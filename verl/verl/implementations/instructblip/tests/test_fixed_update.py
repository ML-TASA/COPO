from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verl.implementations.instructblip.alignment import append_response_tokens
from verl.implementations.instructblip.config import InstructBlipConfig
from verl.implementations.instructblip.processor import (
    SPLIT_CONTEXT_TWO_SHOT_PROTOCOL,
    render_copo_prompt,
    render_prompt_context,
)
from verl.implementations.instructblip.train import (
    InstructBlipCOPOTrainer,
    load_samples,
    parse_args,
    validate_store_protocol,
)
from verl.utils.causal_reward.tests.synthetic_v3 import make_synthetic_v3_store
from verl.utils.causal_reward.trajectory import FixedTrajectoryStore


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.seen_actions = []
        self.seen_instructions = []
        self.seen_qformer_instructions = []

    def process_response_ids(
        self, image, instruction, response_ids, *, qformer_instruction=None
    ):
        self.seen_instructions.append(instruction)
        self.seen_qformer_instructions.append(qformer_instruction)
        token_ids = torch.as_tensor(response_ids).clone().long()
        self.seen_actions.append(token_ids.tolist())
        input_ids, attention_mask, labels = append_response_tokens(
            torch.tensor([[11, 12]]), torch.ones(1, 2, dtype=torch.long), token_ids, 6
        )
        return {
            "pixel_values": torch.zeros(1, 3, 2, 2),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "qformer_input_ids": torch.ones(1, 1, dtype=torch.long),
            "qformer_attention_mask": torch.ones(1, 1, dtype=torch.long),
            "labels": labels,
        }

    def collate(self, samples):
        result = {"pixel_values": torch.cat([item["pixel_values"] for item in samples])}
        for key, pad in (("input_ids", 0), ("attention_mask", 0), ("labels", -100)):
            width = max(item[key].shape[1] for item in samples)
            result[key] = torch.cat(
                [torch.nn.functional.pad(item[key], (0, width - item[key].shape[1]), value=pad) for item in samples]
            )
        for key in ("qformer_input_ids", "qformer_attention_mask"):
            result[key] = torch.cat([item[key] for item in samples])
        return result

    def generate_inputs(self, image, instruction, *, qformer_instruction=None):
        return {"input_ids": torch.tensor([[11, 12]])}

    def decode(self, token_ids):
        return "correct" if int(token_ids[0]) == 21 else "wrong"


class _TinyInstructBlip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_logits = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 128))
        self.frozen_offset = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)

    @property
    def device(self):
        return self.policy_logits.device

    def forward(self, input_ids, **kwargs):
        batch, length = input_ids.shape
        logits = (self.policy_logits + self.frozen_offset).reshape(1, 1, -1).expand(batch, length, -1)
        return SimpleNamespace(logits=logits)


class _Outcome:
    def compute(self, responses, answers=None, tasks=None):
        assert answers == ["yes", "yes"]
        return torch.tensor([1.0 if text == "correct" else 0.0 for text in responses])


def test_instructblip_fixed_v3_eq11_performs_optimizer_step_without_generation(monkeypatch):
    from verl.implementations.instructblip import train as train_module

    assert not hasattr(train_module, "generate_responses")
    store_value = make_synthetic_v3_store()
    store_value["metadata"]["model_family"] = "instructblip"
    store = FixedTrajectoryStore(store_value)
    model = _TinyInstructBlip()
    processor = _Processor()
    config = InstructBlipConfig(
        model_name_or_path="synthetic-fixture",
        batch_size=2,
        max_txt_len=6,
        lr=0.05,
        weight_decay=0.0,
        clip_ratio=0.2,
        kl_loss_coef=0.01,
    )
    monkeypatch.setattr(
        "verl.implementations.instructblip.train.multimodal_content_hash", lambda value: "d" * 64
    )
    trainer = InstructBlipCOPOTrainer(
        model,
        config,
        processor,
        _Outcome(),
        fixed_store=store,
        answers=["yes"],
        tasks=["vqa"],
        sample_ids=["sample-0"],
        group_ids=["group-0"],
        image_ids=["image-0"],
    )
    before = model.policy_logits.detach().clone()
    frozen_before = model.frozen_offset.detach().clone()
    loss = trainer.step([object()], ["question"])
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(model.policy_logits.detach(), before)
    assert torch.equal(model.frozen_offset.detach(), frozen_before)
    assert processor.seen_actions == [[21, 101, 31, 102, 2], [22, 101, 32, 102, 2]]
    assert processor.seen_instructions == [render_copo_prompt("question")] * 2
    assert processor.seen_qformer_instructions == [render_copo_prompt("question")] * 2


def test_instructblip_optional_split_context_fixed_update_uses_same_stage_context(monkeypatch):
    store_value = make_synthetic_v3_store()
    store_value["metadata"]["model_family"] = "instructblip"
    store_value["metadata"]["sampling_parameters"]["protocol_id"] = (
        SPLIT_CONTEXT_TWO_SHOT_PROTOCOL
    )
    store = FixedTrajectoryStore(store_value)
    model = _TinyInstructBlip()
    processor = _Processor()
    config = InstructBlipConfig(
        model_name_or_path="synthetic-fixture",
        batch_size=2,
        max_txt_len=6,
        lr=0.05,
        weight_decay=0.0,
        generation_protocol=SPLIT_CONTEXT_TWO_SHOT_PROTOCOL,
    )
    monkeypatch.setattr(
        "verl.implementations.instructblip.train.multimodal_content_hash",
        lambda value: "d" * 64,
    )
    trainer = InstructBlipCOPOTrainer(
        model,
        config,
        processor,
        _Outcome(),
        fixed_store=store,
        answers=["yes"],
        tasks=["vqa"],
        sample_ids=["sample-0"],
        group_ids=["group-0"],
        image_ids=["image-0"],
    )

    loss = trainer.step([object()], ["question"])

    llm_prompt, qformer_prompt = render_prompt_context(
        "question", SPLIT_CONTEXT_TWO_SHOT_PROTOCOL
    )
    assert torch.isfinite(torch.tensor(loss))
    assert processor.seen_instructions == [llm_prompt] * 2
    assert processor.seen_qformer_instructions == [qformer_prompt] * 2
    assert processor.seen_actions == [[21, 101, 31, 102, 2], [22, 101, 32, 102, 2]]


def test_instructblip_original_and_candidate_stores_cannot_be_mixed():
    original = FixedTrajectoryStore(make_synthetic_v3_store())
    candidate_value = make_synthetic_v3_store()
    candidate_value["metadata"]["sampling_parameters"]["protocol_id"] = (
        SPLIT_CONTEXT_TWO_SHOT_PROTOCOL
    )
    candidate = FixedTrajectoryStore(candidate_value)

    validate_store_protocol(original, "original")
    validate_store_protocol(candidate, SPLIT_CONTEXT_TWO_SHOT_PROTOCOL)
    with pytest.raises(ValueError, match="generation protocol mismatch"):
        validate_store_protocol(original, SPLIT_CONTEXT_TWO_SHOT_PROTOCOL)
    with pytest.raises(ValueError, match="generation protocol mismatch"):
        validate_store_protocol(candidate, "original")


def test_instructblip_fixed_update_rejects_effective_prompt_mismatch(monkeypatch):
    store_value = make_synthetic_v3_store()
    store_value["metadata"]["model_family"] = "instructblip"
    store = FixedTrajectoryStore(store_value)
    model = _TinyInstructBlip()
    processor = _Processor()
    processor.generate_inputs = lambda image, instruction, **kwargs: {
        "input_ids": torch.tensor([[11, 99]])
    }
    config = InstructBlipConfig(
        model_name_or_path="synthetic-fixture", batch_size=2, max_txt_len=6
    )
    monkeypatch.setattr(
        "verl.implementations.instructblip.train.multimodal_content_hash", lambda value: "d" * 64
    )
    trainer = InstructBlipCOPOTrainer(
        model,
        config,
        processor,
        _Outcome(),
        fixed_store=store,
        answers=["yes"],
        tasks=["vqa"],
        sample_ids=["sample-0"],
        group_ids=["group-0"],
        image_ids=["image-0"],
    )
    with pytest.raises(ValueError, match="prompt token mismatch"):
        trainer.step([object()], ["question"])


def test_instructblip_rollout_keeps_sampled_eos_action():
    from verl.implementations.instructblip.rollout import _trim_response

    assert _trim_response([7, 2, 0, 0], eos_id=2, pad_id=0, keep_eos=True) == [7, 2]
    assert _trim_response([7, 2, 0, 0], eos_id=2, pad_id=0) == [7]


def test_instructblip_fixed_dataset_requires_explicit_group_and_image_identity(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text('{"sample_id":"s","image":"image.jpg","instruction":"q","answer":"a"}\n')
    with pytest.raises(ValueError, match="group_id"):
        load_samples(str(dataset), fixed_trajectory=True)


def test_instructblip_public_cli_requires_fixed_trajectory():
    with pytest.raises(SystemExit):
        parse_args([
            "--model", "model",
            "--dataset", "samples.jsonl",
            "--causal-reward", "trajectories.pt",
            "--steps", "1",
        ])


def test_instructblip_public_cli_keeps_original_protocol_default():
    args = parse_args([
        "--model", "model",
        "--dataset", "samples.jsonl",
        "--causal-reward", "trajectories.pt",
        "--fixed-trajectory",
        "--steps", "1",
    ])
    assert args.instructblip_protocol == "original"


def test_instructblip_trainer_rejects_non_eq11_rollout_mode():
    trainer = InstructBlipCOPOTrainer(
        _TinyInstructBlip(),
        InstructBlipConfig(model_name_or_path="synthetic-fixture", batch_size=2, max_txt_len=6),
        _Processor(),
        _Outcome(),
    )
    with pytest.raises(RuntimeError, match="requires a validated.*v3 store"):
        trainer.step([], [])


def test_instructblip_public_cli_rejects_dummy_outcome_reward():
    with pytest.raises(SystemExit):
        parse_args([
            "--model", "model",
            "--dataset", "samples.jsonl",
            "--causal-reward", "trajectories.pt",
            "--fixed-trajectory",
            "--steps", "1",
            "--outcome-reward", "dummy",
        ])
