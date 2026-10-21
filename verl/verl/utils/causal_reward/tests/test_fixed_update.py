import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop import FixedAgentLoopProcessor
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.trainer import main_ppo
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss, kl_penalty
from verl.trainer.ppo.ray_trainer import (
    compute_advantage,
    fixed_trajectory_group_uids,
    resume_needs_dataloader_state,
    select_fixed_trajectory_batch,
)
from verl.trainer.ppo.utils import Role
from verl.utils.causal_reward.integration import COPOConfig
from verl.utils.causal_reward.optimization import compute_copo_eq11_policy_loss
from verl.utils.causal_reward.tests.synthetic_v3 import make_synthetic_v3_store
from verl.utils.causal_reward.trajectory import FixedTrajectoryStore, make_trajectory_id


class TinyPolicy(torch.nn.Module):
    def __init__(self, vocab_size=128, hidden_size=12):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def action_log_probs(self, prompts, responses, response_mask):
        first_context = prompts[:, -1:]
        contexts = torch.cat([first_context, responses[:, :-1]], dim=1)
        logits = self.output(self.embedding(contexts))
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)
        return log_probs * response_mask


def _worker_role_config(fixed: bool):
    return OmegaConf.create(
        {
            "trainer": {"use_legacy_worker_impl": "auto"},
            "actor_rollout_ref": {"actor": {"strategy": "fsdp"}},
            "algorithm": {"name": "copo", "copo": {"fixed_trajectory": fixed}},
        }
    )


def test_fixed_mode_registers_actor_only_while_online_keeps_rollout(monkeypatch):
    monkeypatch.setattr(main_ppo.ray, "remote", lambda worker: worker)

    fixed_runner = main_ppo.TaskRunner()
    fixed_runner.add_actor_rollout_worker(_worker_role_config(True))
    assert set(fixed_runner.role_worker_mapping) == {Role.Actor}

    online_runner = main_ppo.TaskRunner()
    online_runner.add_actor_rollout_worker(_worker_role_config(False))
    assert set(online_runner.role_worker_mapping) == {Role.ActorRollout}


def test_fixed_processor_has_no_generation_server():
    config = OmegaConf.create(
        {
            "data": {},
            "actor_rollout_ref": {
                "model": {"custom_chat_template": None},
                "rollout": {
                    "agent": {"agent_loop_config_path": None, "default_agent_loop": "single_turn_agent"}
                },
            },
        }
    )
    processor = FixedAgentLoopProcessor(config, tokenizer=object(), processor=object())
    assert processor.server_manager is None


def test_resume_restores_dataloader_only_inside_an_epoch():
    assert resume_needs_dataloader_state(global_steps=3, dataloader_length=8)
    assert not resume_needs_dataloader_state(global_steps=8, dataloader_length=8)
    assert not resume_needs_dataloader_state(global_steps=16, dataloader_length=8)
    with pytest.raises(ValueError, match="positive"):
        resume_needs_dataloader_state(global_steps=1, dataloader_length=0)


def _native_data_proto(store):
    prompts = torch.tensor([[0, 11, 12], [0, 11, 12]], dtype=torch.long)
    responses = store["response_token_ids"].clone()
    prompt_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.long)
    response_attention = store["response_mask"].clone()
    attention = torch.cat([prompt_mask, response_attention], dim=-1)
    input_ids = torch.cat([prompts, responses], dim=-1)
    position_ids = attention.cumsum(-1) - 1
    position_ids.masked_fill_(attention == 0, 0)
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_attention,
            "attention_mask": attention,
            "input_ids": input_ids,
            "position_ids": position_ids,
        },
        batch_size=2,
    )
    multi_modal = np.empty(2, dtype=object)
    multi_modal[:] = [{"pixel_values": torch.ones(1)}, {"pixel_values": torch.ones(1)}]
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "sample_id": np.array(["sample-0", "sample-0"], dtype=object),
            "group_id": np.array(["group-0", "group-0"], dtype=object),
            "image_id": np.array(["image-0", "image-0"], dtype=object),
            "image_hash": np.array(["d" * 64, "d" * 64], dtype=object),
            "copo_actual_image_hash": np.array(["d" * 64, "d" * 64], dtype=object),
            "uid": np.array(["prompt-0", "prompt-0"], dtype=object),
            "multi_modal_inputs": multi_modal,
        },
    )


def _bind_policy_log_probs(store, old_log_probs, reference_log_probs):
    mask = store["response_mask"].bool()
    store["old_policy_log_probs"] = old_log_probs.detach() * mask
    store["reference_log_probs"] = reference_log_probs.detach() * mask
    lengths = store["response_lengths"]
    metadata = store["metadata"]
    store["trajectory_ids"] = [
        make_trajectory_id(
            sample_id=store["sample_ids"][row],
            group_id=store["group_ids"][row],
            collection_round=int(metadata["collection_round"]),
            rollout_index=int(store["rollout_indices"][row]),
            prompt_hash=store["prompt_hashes"][row],
            image_hash=store["image_hashes"][row],
            response_token_ids=store["response_token_ids"][row, : lengths[row]].tolist(),
            response_mask=store["response_mask"][row, : lengths[row]].tolist(),
            reasoning_mask=store["reasoning_mask"][row, : lengths[row]].tolist(),
            answer_mask=store["answer_mask"][row, : lengths[row]].tolist(),
            special_token_mask=store["special_token_mask"][row, : lengths[row]].tolist(),
            collector_policy_id=metadata["collector_policy_id"],
            reference_policy_id=metadata["reference_policy_id"],
            old_policy_log_probs=store["old_policy_log_probs"][row, : lengths[row]].tolist(),
            reference_log_probs=store["reference_log_probs"][row, : lengths[row]].tolist(),
            outcome_reward=float(store["outcome_rewards"][row]),
            sufficiency=store["sufficiency"][row, : lengths[row]].tolist(),
            necessity=store["necessity"][row, : lengths[row]].tolist(),
            causal_reward=store["causal_reward"][row, : lengths[row]].tolist(),
        )
        for row in range(2)
    ]


def test_native_fixed_mapping_preserves_multimodal_identity_and_tokens():
    store_value = make_synthetic_v3_store()
    store = FixedTrajectoryStore(store_value)
    original = DataProto(
        batch=TensorDict({"dummy": torch.zeros(1, 1)}, batch_size=1),
        non_tensor_batch={
            "sample_id": np.array(["sample-0"], dtype=object),
            "group_id": np.array(["group-0"], dtype=object),
            "image_id": np.array(["image-0"], dtype=object),
            "image_hash": np.array(["d" * 64], dtype=object),
        },
    )
    selected = select_fixed_trajectory_batch(store, original)
    assert fixed_trajectory_group_uids(original).tolist() == ['["sample-0","group-0"]']
    generation_input = original.repeat(repeat_times=2, interleave=True)
    selected.add_generation_inputs(generation_input)
    assert generation_input.non_tensor_batch["copo_fixed_response_token_ids"].tolist() == [
        [21, 101, 31, 102, 2],
        [22, 101, 32, 102, 2],
    ]
    assert generation_input.non_tensor_batch["copo_fixed_prompt_suffix_token_ids"].tolist() == [[], []]
    data = _native_data_proto(store_value)
    selected.attach_to_dataproto(data)
    assert torch.equal(data.batch["responses"], store_value["response_token_ids"])
    assert torch.equal(data.batch["old_log_probs"], store_value["old_policy_log_probs"])
    assert data.non_tensor_batch["image_id"].tolist() == ["image-0", "image-0"]
    assert all("pixel_values" in item for item in data.non_tensor_batch["multi_modal_inputs"])


def test_native_mapping_rejects_changed_action_or_prompt_tokens():
    store_value = make_synthetic_v3_store()
    selected = FixedTrajectoryStore(store_value).select_groups(
        ["sample-0"], ["group-0"], ["image-0"], ["d" * 64]
    )
    changed_action = _native_data_proto(store_value)
    changed_action.batch["responses"][0, 0] = 99
    with pytest.raises(ValueError, match="response tokens changed"):
        selected.attach_to_dataproto(changed_action)

    changed_prompt = _native_data_proto(store_value)
    changed_prompt.batch["prompts"][0, -1] = 99
    with pytest.raises(ValueError, match="prompt token mismatch"):
        selected.attach_to_dataproto(changed_prompt)

    changed_image = _native_data_proto(store_value)
    changed_image.non_tensor_batch["copo_actual_image_hash"][0] = "e" * 64
    with pytest.raises(ValueError, match="processed image hash mismatch"):
        selected.attach_to_dataproto(changed_image)


def test_fixed_response_agent_loop_never_calls_generation_server():
    loop = object.__new__(SingleTurnAgentLoop)
    loop.response_length = 6
    loop.tool_schemas = []
    loop.process_vision_info = AsyncMock(return_value={"images": ["synthetic-image"], "videos": None})
    loop.apply_chat_template = AsyncMock(return_value=[11, 12])
    loop.server_manager = type(
        "Server", (), {"generate": AsyncMock(side_effect=AssertionError("generation called"))}
    )()
    loop.tokenizer = type("Tokenizer", (), {"decode": lambda self, ids, skip_special_tokens: "fixed"})()
    output = asyncio.run(
        loop.run(
            {},
            raw_prompt=[],
            copo_fixed_response_token_ids=[21, 101, 31, 102, 2],
            copo_fixed_outcome_reward=1.0,
        )
    )
    assert output.response_ids == [21, 101, 31, 102, 2]
    assert output.reward_score == 1.0
    assert len(output.extra_fields["copo_actual_image_hash"]) == 64
    assert loop.server_manager.generate.await_count == 0


def test_fixed_response_agent_loop_replays_stored_prompt_suffix_without_generation():
    loop = object.__new__(SingleTurnAgentLoop)
    loop.response_length = 6
    loop.tool_schemas = []
    loop.process_vision_info = AsyncMock(return_value={"images": ["synthetic-image"], "videos": None})
    loop.apply_chat_template = AsyncMock(return_value=[11, 12])
    loop.server_manager = type(
        "Server", (), {"generate": AsyncMock(side_effect=AssertionError("generation called"))}
    )()
    loop.tokenizer = type("Tokenizer", (), {"decode": lambda self, ids, skip_special_tokens: "fixed"})()
    output = asyncio.run(
        loop.run(
            {},
            raw_prompt=[],
            copo_fixed_prompt_suffix_token_ids=[10],
            copo_fixed_response_token_ids=[21, 101, 31, 102, 2],
            copo_fixed_outcome_reward=1.0,
        )
    )
    assert output.prompt_ids == [11, 12, 10]
    assert output.response_ids == [21, 101, 31, 102, 2]
    assert loop.server_manager.generate.await_count == 0


def test_fixed_llava_agent_loop_hashes_unresized_collector_image(tmp_path):
    from PIL import Image
    from verl.utils.causal_reward.trajectory import multimodal_content_hash

    image_path = tmp_path / "image.png"
    image = Image.new("RGB", (17, 11), color=(1, 2, 3))
    image.save(image_path)
    loop = object.__new__(SingleTurnAgentLoop)
    loop.response_length = 2
    loop.tool_schemas = []
    loop.config = SimpleNamespace(
        algorithm=SimpleNamespace(copo=SimpleNamespace(fixed_model_family="llava15"))
    )
    loop.process_vision_info = AsyncMock(side_effect=AssertionError("legacy fixed image was resized"))
    loop.apply_chat_template = AsyncMock(return_value=[11])
    loop.server_manager = type(
        "Server", (), {"generate": AsyncMock(side_effect=AssertionError("generation called"))}
    )()
    loop.tokenizer = type("Tokenizer", (), {"decode": lambda self, ids, skip_special_tokens: "fixed"})()
    output = asyncio.run(
        loop.run(
            {},
            raw_prompt=[{"role": "user", "content": [
                {"type": "image", "image": image_path.as_uri()}, {"type": "text", "text": "question"}
            ]}],
            copo_fixed_response_token_ids=[2],
            copo_fixed_outcome_reward=1.0,
        )
    )
    expected = multimodal_content_hash({"images": [image], "videos": None})
    assert output.extra_fields["copo_actual_image_hash"] == expected
    assert output.multi_modal_data["images"][0].size == (17, 11)


def test_tiny_fixed_trajectory_completes_real_optimizer_step():
    torch.manual_seed(7)
    current = TinyPolicy()
    old_policy = TinyPolicy()
    reference = TinyPolicy()
    old_policy.load_state_dict(current.state_dict())
    reference.load_state_dict(current.state_dict())
    for model in (old_policy, reference):
        model.requires_grad_(False)

    store_value = make_synthetic_v3_store()
    prompts = torch.tensor([[0, 11, 12], [0, 11, 12]], dtype=torch.long)
    responses = store_value["response_token_ids"]
    response_mask = store_value["response_mask"].float()
    with torch.no_grad():
        old_log_probs = old_policy.action_log_probs(prompts, responses, response_mask)
        reference_log_probs = reference.action_log_probs(prompts, responses, response_mask)
    _bind_policy_log_probs(store_value, old_log_probs, reference_log_probs)

    store = FixedTrajectoryStore(store_value)
    data = _native_data_proto(store_value)
    selected = store.select_groups(["sample-0"], ["group-0"], ["image-0"], ["d" * 64])
    selected.attach_to_dataproto(data)

    config = {"name": "copo", "copo": asdict(COPOConfig(enable=True, reward_source="precomputed"))}
    data.batch["token_level_rewards"] = data.batch["rm_scores"]
    data = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config=config,
    )
    causal = store_value["causal_reward"]
    reasoning = store_value["reasoning_mask"].bool()
    answer = store_value["answer_mask"].bool()
    outcome = store_value["outcome_rewards"]
    group_only = (outcome - outcome.mean()) / (outcome.std() + 1.0e-6)
    assert torch.allclose(
        data.batch["advantages"][reasoning] - causal[reasoning],
        group_only,
        atol=1e-5,
    )
    assert torch.allclose(data.batch["advantages"][answer], group_only, atol=1e-5)
    assert torch.equal(data.batch["copo_causal_rewards"][answer], torch.zeros_like(causal[answer]))
    assert torch.equal(
        data.batch["copo_causal_rewards"][store_value["special_token_mask"].bool()],
        torch.zeros_like(causal[store_value["special_token_mask"].bool()]),
    )

    optimizer = torch.optim.SGD(current.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in current.parameters()]
    stored_old = data.batch["old_log_probs"].clone()
    current_log_probs = current.action_log_probs(prompts, responses, response_mask)
    assert torch.equal(data.batch["old_log_probs"], old_log_probs)
    policy_loss, _ = compute_copo_eq11_policy_loss(
        data.batch["old_log_probs"],
        current_log_probs,
        data.batch["advantages"],
        data.batch["response_mask"],
        0.2,
        0.2,
    )
    kld = kl_penalty(current_log_probs, data.batch["ref_log_prob"], "low_var_kl")
    kl_loss = agg_loss(kld, data.batch["response_mask"], "seq-mean-token-mean")
    loss = policy_loss + 0.001 * kl_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, current.parameters(), strict=True))
    updated_log_probs = current.action_log_probs(prompts, responses, response_mask)
    assert torch.equal(data.batch["old_log_probs"], stored_old)
    assert not torch.allclose(updated_log_probs[response_mask.bool()], stored_old[response_mask.bool()])
    assert all(parameter.grad is None for parameter in old_policy.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())
