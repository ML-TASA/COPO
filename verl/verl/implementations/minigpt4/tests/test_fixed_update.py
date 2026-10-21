from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import pytest

from verl.implementations.minigpt4.config import MiniGPT4Config
from verl.implementations.minigpt4.mini_gpt4 import MiniGPT4
from verl.implementations.minigpt4.trainer import COPOTrainer
from verl.implementations.minigpt4.train import load_samples, parse_args
from verl.utils.causal_reward.tests.synthetic_v3 import make_synthetic_v3_store
from verl.utils.causal_reward.trajectory import (
    FixedTrajectoryStore,
    make_trajectory_id,
    token_sequence_hash,
)


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "left"

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False, **kwargs):
        self.calls.append(text)
        token = 11 if str(text).startswith("before") else 12
        return SimpleNamespace(input_ids=[token])

    def decode(self, token_ids, skip_special_tokens=True):
        return "correct" if int(token_ids[0]) == 21 else "wrong"

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.decode(row.tolist(), skip_special_tokens) for row in token_ids]


class _FrozenLM(torch.nn.Module):
    def __init__(self, hidden=4, vocab=128):
        super().__init__()
        self.output = torch.nn.Linear(hidden, vocab, bias=False)
        self.output.requires_grad_(False)
        self.config = SimpleNamespace(max_position_embeddings=128)

    def forward(self, inputs_embeds, attention_mask, return_dict=True):
        states = inputs_embeds.cumsum(dim=1)
        return SimpleNamespace(logits=self.output(states))


class _TinyMiniGPT4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llama_proj = torch.nn.Linear(1, 4, bias=False)
        self.token_embedding = torch.nn.Embedding(128, 4)
        self.token_embedding.requires_grad_(False)
        self.llama_model = _FrozenLM()
        self.llama_tokenizer = _Tokenizer()
        self.max_txt_len = 6
        self.end_sym = "###"

    @property
    def device(self):
        return self.llama_proj.weight.device

    def encode_img(self, image):
        scalar = image.mean(dim=(1, 2, 3), keepdim=True).reshape(image.shape[0], 1, 1)
        embeds = self.llama_proj(scalar)
        return embeds, torch.ones(image.shape[0], 1, dtype=torch.long, device=image.device)

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        return img_embeds, atts_img, 0

    def embed_tokens(self, ids):
        return self.token_embedding(ids)

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        values, masks, lengths = [], [], []
        for row in range(input_embs.shape[0]):
            length = int(input_atts[row].sum())
            lengths.append(torch.tensor(length))
            values.append(torch.cat([input_embs[row, :length], output_embs[row], input_embs[row, length:]]))
            masks.append(torch.cat([input_atts[row, :length], output_atts[row], input_atts[row, length:]]))
        return torch.stack(values), torch.stack(masks), lengths

    def maybe_autocast(self):
        return nullcontext()


class _Processor:
    def preprocess_image(self, image):
        return torch.ones(3, 2, 2)

    def build_prompt(self, instruction):
        return "before<ImageHere>after"


class _Outcome:
    def compute(self, responses, answers=None, tasks=None):
        assert answers == ["yes", "yes"]
        return torch.tensor([1.0 if text == "correct" else 0.0 for text in responses])


def _mini_store():
    store = make_synthetic_v3_store()
    store["metadata"]["model_family"] = "minigpt4"
    store["prompt_token_ids"][:] = torch.tensor([1, 11, 12])
    store["prompt_lengths"][:] = 3
    prompt_hash = token_sequence_hash([1, 11, 12])
    store["prompt_hashes"] = [prompt_hash, prompt_hash]
    store["trajectory_ids"] = [
        make_trajectory_id(
            sample_id=store["sample_ids"][row],
            group_id=store["group_ids"][row],
            collection_round=0,
            rollout_index=int(store["rollout_indices"][row]),
            prompt_hash=prompt_hash,
            image_hash=store["image_hashes"][row],
            response_token_ids=store["response_token_ids"][row, :5].tolist(),
            response_mask=store["response_mask"][row, :5].tolist(),
            reasoning_mask=store["reasoning_mask"][row, :5].tolist(),
            answer_mask=store["answer_mask"][row, :5].tolist(),
            special_token_mask=store["special_token_mask"][row, :5].tolist(),
            collector_policy_id=store["metadata"]["collector_policy_id"],
            reference_policy_id=store["metadata"]["reference_policy_id"],
            old_policy_log_probs=store["old_policy_log_probs"][row, :5].tolist(),
            reference_log_probs=store["reference_log_probs"][row, :5].tolist(),
            outcome_reward=float(store["outcome_rewards"][row]),
            sufficiency=store["sufficiency"][row, :5].tolist(),
            necessity=store["necessity"][row, :5].tolist(),
            causal_reward=store["causal_reward"][row, :5].tolist(),
        )
        for row in range(2)
    ]
    return store


def test_minigpt4_fixed_v3_eq11_uses_exact_ids_and_updates_only_projection(monkeypatch):
    from verl.implementations.minigpt4 import trainer as trainer_module

    assert not hasattr(trainer_module, "generate_group")
    model = _TinyMiniGPT4()
    config = MiniGPT4Config(
        llama_model="synthetic-fixture",
        group_size=2,
        grpo_max_new_tokens=6,
        max_txt_len=6,
        lr=0.05,
        weight_decay=0.0,
        clip_ratio=0.2,
        kl_loss_coef=0.01,
    )
    monkeypatch.setattr(
        "verl.implementations.minigpt4.trainer.multimodal_content_hash", lambda value: "d" * 64
    )
    trainer = COPOTrainer(
        model,
        config,
        _Processor(),
        _Outcome(),
        fixed_store=FixedTrajectoryStore(_mini_store()),
        answers=["yes"],
        tasks=["vqa"],
        sample_ids=["sample-0"],
        group_ids=["group-0"],
        image_ids=["image-0"],
    )
    projection_before = model.llama_proj.weight.detach().clone()
    embedding_before = model.token_embedding.weight.detach().clone()
    lm_before = model.llama_model.output.weight.detach().clone()
    stats = trainer.step([object()], ["question"])
    assert torch.isfinite(torch.tensor(stats["loss"]))
    assert stats["grad_norm"] > 0
    assert not torch.equal(model.llama_proj.weight.detach(), projection_before)
    assert torch.equal(model.token_embedding.weight.detach(), embedding_before)
    assert torch.equal(model.llama_model.output.weight.detach(), lm_before)
    assert all("###" not in call for call in model.llama_tokenizer.calls)


class _GenerateLM:
    def __init__(self, output):
        self.output = output

    def generate(self, **kwargs):
        return self.output


class _GenerateFixture:
    generate = MiniGPT4.generate

    def __init__(self, output):
        self.llama_tokenizer = _Tokenizer()
        self.llama_model = _GenerateLM(output)
        self.prompt_list = []

    def encode_img(self, image):
        return torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.long)

    def prompt_wrap(self, img, mask, prompt):
        return img, mask, 0

    def embed_tokens(self, ids):
        return torch.zeros(*ids.shape, 4)

    def maybe_autocast(self):
        return nullcontext()


def test_generation_returns_raw_actions_without_mutating_output_tensor():
    generated = torch.tensor([[21, 31, 2, 0], [22, 32, 2, 0]])
    original = generated.clone()
    fixture = _GenerateFixture(generated)
    texts, actions = fixture.generate(
        {"image": torch.zeros(1, 3, 2, 2), "prompt": "before<ImageHere>after"},
        num_captions=2,
        return_token_ids=True,
    )
    assert torch.equal(generated, original)
    assert [action.tolist() for action in actions] == [[21, 31, 2], [22, 32, 2]]
    assert len(texts) == 2


def test_minigpt4_fixed_dataset_requires_explicit_group_and_image_identity(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text('{"sample_id":"s","image":"image.jpg","instruction":"q","answer":"a"}\n')
    with pytest.raises(ValueError, match="group_id"):
        load_samples(str(dataset), fixed_trajectory=True)


def test_minigpt4_public_cli_requires_fixed_trajectory():
    with pytest.raises(SystemExit):
        parse_args([
            "--llama-model", "model",
            "--dataset", "samples.jsonl",
            "--causal-reward", "trajectories.pt",
            "--grpo-steps", "1",
        ])


def test_minigpt4_trainer_rejects_non_eq11_rollout_mode():
    model = _TinyMiniGPT4()
    config = MiniGPT4Config(
        llama_model="fixture",
        vit_pretrained_path="fixture",
        q_former_model="fixture",
        ckpt="fixture",
        bert_model_path="fixture",
        group_size=2,
        grpo_max_new_tokens=6,
        max_txt_len=6,
    )
    trainer = COPOTrainer(model, config, _Processor(), _Outcome())
    with pytest.raises(RuntimeError, match="requires a validated.*v3 store"):
        trainer.step([], [])


def test_minigpt4_public_cli_rejects_dummy_outcome_reward():
    with pytest.raises(SystemExit):
        parse_args([
            "--llama-model", "model",
            "--dataset", "samples.jsonl",
            "--causal-reward", "trajectories.pt",
            "--fixed-trajectory",
            "--grpo-steps", "1",
            "--outcome-reward", "dummy",
        ])
