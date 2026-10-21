from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from verl.utils.causal_reward.backends import build_backend
from verl.utils.causal_reward.backends.llava15 import Llava15CollectorBackend
from verl.utils.causal_reward.backends.hf_decoder import HFDecoderCollectorBackend, generation_eos_token_ids
from verl.utils.causal_reward.collector import CollectorIdentity
from verl.implementations.minigpt4.config import MiniGPT4Config
from verl.implementations.instructblip.processor import (
    ORIGINAL_FORMAT_INSTRUCTION,
    SPLIT_CONTEXT_TWO_SHOT_PROTOCOL,
    THINK_PREFILL,
    protocol_processor_fingerprint,
    render_copo_prompt,
    render_prompt_context,
)
from verl.utils.causal_reward.generator import load_collection_samples, main, parse_args
from verl.utils.causal_reward.trajectory import fingerprint_processor_template


@pytest.mark.parametrize(
    ("family", "module_name", "class_name"),
    [
        ("llava15", "llava15", "Llava15CollectorBackend"),
        ("qwenvl", "qwen_vl", "QwenVLCollectorBackend"),
        ("instructblip", "instructblip", "InstructBlipCollectorBackend"),
        ("minigpt4", "minigpt4", "MiniGPT4CollectorBackend"),
    ],
)
def test_backend_dispatch_is_model_specific(monkeypatch, family, module_name, class_name):
    module = importlib.import_module(f"verl.utils.causal_reward.backends.{module_name}")
    calls = []

    class Sentinel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, class_name, Sentinel)
    result = build_backend(family, checkpoint="local", device="cpu")
    assert isinstance(result, Sentinel)
    assert calls == [{"checkpoint": "local", "device": "cpu"}]


def test_backend_dispatch_rejects_unknown_family():
    with pytest.raises(ValueError, match="unsupported"):
        build_backend("qwen2_5_vl", checkpoint="unused")


def test_llava_missing_checkpoint_fails_before_model_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist locally"):
        Llava15CollectorBackend(str(tmp_path / "missing"), device="cpu")


def test_llava_collector_uses_native_verl_fast_processor(monkeypatch, tmp_path):
    module = importlib.import_module("verl.utils.causal_reward.backends.llava15")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls = []

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            raise RuntimeError("stop after processor construction")

    monkeypatch.setattr(module, "AutoProcessor", ProcessorFactory)
    with pytest.raises(RuntimeError, match="stop after processor construction"):
        module.Llava15CollectorBackend(str(checkpoint), device="cpu")
    assert calls == [(str(checkpoint), {"use_fast": True})]


def test_instructblip_collector_uses_prompt_prefill_protocol():
    module = importlib.import_module("verl.utils.causal_reward.backends.instructblip")
    backend = module.InstructBlipCollectorBackend.__new__(module.InstructBlipCollectorBackend)
    assert backend.reasoning_prefill is True
    rendered = backend.render_prompt("Is there a television in the image?")
    assert rendered == "Is there a television in the image?\nAssistant: <think>"
    assert rendered.endswith(THINK_PREFILL)
    assert "<image>" not in rendered
    assert backend.render_prompt(rendered + "  ") == rendered


def test_instructblip_prompt_rejects_manual_image_placeholder():
    with pytest.raises(ValueError, match="must not contain a literal <image>"):
        render_copo_prompt("User: <image> What is shown?")


def test_instructblip_split_context_is_explicit_and_does_not_change_original():
    question = "Is there a television in the image?"
    canonical = question + ORIGINAL_FORMAT_INSTRUCTION
    original_llm, original_qformer = render_prompt_context(canonical, "original")
    split_llm, split_qformer = render_prompt_context(
        canonical, SPLIT_CONTEXT_TWO_SHOT_PROTOCOL
    )

    assert original_llm == render_copo_prompt(canonical)
    assert original_qformer == original_llm
    assert split_qformer == question
    assert split_llm.endswith("Assistant: <think>")
    assert split_llm.count("Current question: " + question) == 1
    assert "<answer>yes</answer>" in split_llm
    assert "<answer>no</answer>" in split_llm
    assert "<image>" not in split_llm
    assert canonical not in split_llm


def test_instructblip_split_context_rejects_unknown_structural_prompt():
    with pytest.raises(ValueError, match="unformatted task instruction"):
        render_prompt_context(
            "Question with <think>custom markup",
            SPLIT_CONTEXT_TWO_SHOT_PROTOCOL,
        )


def test_instructblip_protocol_fingerprint_preserves_original_and_isolates_candidate():
    processor = SimpleNamespace(
        chat_template=None,
        to_dict=lambda: {"num_query_tokens": 32},
        image_processor=SimpleNamespace(to_dict=lambda: {"size": 224}),
    )
    original = protocol_processor_fingerprint(processor, "original")
    candidate = protocol_processor_fingerprint(
        processor, SPLIT_CONTEXT_TWO_SHOT_PROTOCOL
    )
    assert original == fingerprint_processor_template(processor)
    assert candidate != original


def test_instructblip_split_context_prepares_distinct_llm_and_qformer_inputs():
    module = importlib.import_module("verl.utils.causal_reward.backends.instructblip")
    backend = module.InstructBlipCollectorBackend.__new__(
        module.InstructBlipCollectorBackend
    )
    backend.generation_protocol = SPLIT_CONTEXT_TWO_SHOT_PROTOCOL
    backend.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [len(text)],
        decode=lambda ids, **kwargs: "decoded",
    )
    seen = {}

    class Processor:
        def generate_inputs(self, image, prompt, *, qformer_instruction=None):
            seen["prompt"] = prompt
            seen["qformer"] = qformer_instruction
            return {
                "input_ids": torch.tensor([[99] * 32 + [7, 8]]),
                "attention_mask": torch.ones(1, 34, dtype=torch.long),
                "qformer_input_ids": torch.tensor([[1, 2, 3]]),
                "qformer_attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 2, 2),
            }

    backend.processor = Processor()
    backend._load_image = lambda value: Image.new("RGB", (2, 2), "white")
    prepared = backend.prepare_sample({"image": "unused", "instruction": "Is there a car?"})

    assert prepared["qformer_prompt"] == "Is there a car?"
    assert seen["qformer"] == "Is there a car?"
    assert seen["prompt"] == prepared["prompt"]
    assert prepared["prompt"].endswith("Assistant: <think>")
    assert "<image>" not in prepared["prompt"]
    assert prepared["prompt_token_ids"].count(99) == 32


def test_v3_collection_schema_is_fail_closed(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({
        "sample_id": "s0", "image": "image.png", "prompt": "question", "answer": "yes"
    }) + "\n")
    with pytest.raises(ValueError, match="group_id"):
        load_collection_samples(str(path))


def test_v3_collection_schema_loads_stable_identities(tmp_path):
    path = tmp_path / "data.jsonl"
    image = tmp_path / "image.png"
    image.touch()
    item = {
        "sample_id": "s0", "group_id": "g0", "image_id": "i0", "image": str(image),
        "prompt": "question", "answer": "yes", "task": "vqa",
    }
    path.write_text(json.dumps(item) + "\n")
    loaded = load_collection_samples(str(path))
    assert loaded[0]["instruction"] == "question"
    assert all(loaded[0][key] == value for key, value in item.items())


def test_reward_cli_requires_collector_and_reference_paths():
    with pytest.raises(SystemExit):
        parse_args(["--model", "llava15"])
    args = parse_args([
        "--model", "llava15", "--model-path", "collector", "--reference-model", "reference",
        "--dataset", "data.jsonl", "--output", "store.pt", "--num-rollouts", "2",
        "--max-response-length", "128",
    ])
    assert args.model_name == "llava15"
    assert args.model_path == "collector"
    assert args.reference_model == "reference"
    assert args.num_rollouts == 2
    assert args.max_response_length == 128


def test_reward_cli_exposes_optional_instructblip_protocol_only_for_instructblip():
    args = parse_args([
        "--model", "instructblip", "--model-path", "collector",
        "--reference-model", "reference", "--dataset", "data.jsonl",
        "--output", "store.pt", "--instructblip-protocol", "split_context_two_shot",
    ])
    assert args.instructblip_protocol == "split_context_two_shot"

    with pytest.raises(ValueError, match="only valid with --model instructblip"):
        # Reach the family-specific validation without loading any checkpoint.
        main([
            "--model", "llava15", "--model-path", "missing",
            "--reference-model", "missing", "--dataset", "missing.jsonl",
            "--output", "store.pt", "--instructblip-protocol", "split_context_two_shot",
        ])


def test_peft_collector_state_requires_adapter_config(tmp_path):
    from verl.utils.causal_reward.backends.hf_decoder import load_peft_adapter

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="adapter_config"):
        load_peft_adapter(object(), str(adapter))


def test_minigpt4_cli_uses_base_llm_and_distinct_stage2_checkpoints(monkeypatch, tmp_path):
    backends = importlib.import_module("verl.utils.causal_reward.backends")
    collector_module = importlib.import_module("verl.utils.causal_reward.collector")
    calls = []

    def build(family, **kwargs):
        calls.append((family, kwargs))
        return object()

    class Collector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def collect(self, samples):
            return {"contract": "copo.fixed_trajectory.v3"}

    saved = []
    monkeypatch.setattr(backends, "build_backend", build)
    monkeypatch.setattr(collector_module, "FixedTrajectoryCollector", Collector)
    monkeypatch.setattr(collector_module, "save_fixed_trajectory_store", lambda path, store: saved.append((path, store)))
    data = tmp_path / "data.jsonl"
    image = tmp_path / "image.png"
    image.touch()
    base = tmp_path / "llama-base"
    base.mkdir()
    for name in ("collector-stage2", "reference-stage2", "vit", "qformer", "bert"):
        (tmp_path / name).touch()
    data.write_text(json.dumps({
        "sample_id": "s", "group_id": "g", "image_id": "i", "image": str(image),
        "instruction": "question", "answer": "yes", "task": "vqa",
    }) + "\n")
    result = main([
        "--model", "minigpt4", "--model-path", str(base), "--ckpt", str(tmp_path / "collector-stage2"),
        "--reference-model", str(tmp_path / "reference-stage2"),
        "--vit-pretrained-path", str(tmp_path / "vit"),
        "--qformer-model", str(tmp_path / "qformer"), "--bert-model-path", str(tmp_path / "bert"),
        "--dtype", "bfloat16", "--dataset", str(data),
        "--output", str(tmp_path / "store.pt"),
    ])
    assert result == 0
    assert calls[0][1]["checkpoint"] == str(tmp_path / "collector-stage2")
    assert calls[1][1]["checkpoint"] == str(tmp_path / "reference-stage2")
    assert calls[0][1]["llama_model"] == str(base)
    assert calls[1][1]["llama_model"] == str(base)
    assert calls[0][1]["dtype"] == "bfloat16"
    assert calls[1][1]["dtype"] == "bfloat16"
    assert saved[0][1]["contract"] == "copo.fixed_trajectory.v3"


def test_minigpt4_initial_rollout_reuses_raw_id_generation(monkeypatch):
    module = importlib.import_module("verl.utils.causal_reward.backends.minigpt4")
    backend = module.MiniGPT4CollectorBackend.__new__(module.MiniGPT4CollectorBackend)
    backend.model = object()
    backend.processor = object()
    backend.config = MiniGPT4Config(grpo_max_new_tokens=9)
    backend.identity = CollectorIdentity(
        "minigpt4", "a" * 64, "b" * 64, "c" * 64, "policy", (2,), 0
    )
    calls = []

    def generate(model, config, processor, image, instruction, group_size):
        calls.append((config.grpo_max_new_tokens, image, instruction, group_size))
        raw = torch.tensor([21, 31, 2])
        return [raw, torch.tensor([22, 32])], ["first", "second"]

    monkeypatch.setattr(module, "generate_group_with_ids", generate)
    result = backend.generate_actions(
        {"image": "pil", "instruction": "question"}, num_samples=2, max_new_tokens=7
    )
    assert calls == [(7, "pil", "question", 2)]
    assert [item.token_ids for item in result] == [(21, 31, 2), (22, 32)]
    assert [item.stop_reason for item in result] == ["eos", "length"]
    assert backend.config.grpo_max_new_tokens == 9


class _HFGenerateModel:
    def __init__(self):
        self.inputs = []
        self.generation_kwargs = []
        self.embedded_inputs = []
        self.embedding = torch.nn.Embedding(8, 3)
        self.config = SimpleNamespace(max_position_embeddings=32)
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.generation_kwargs.append(kwargs)
        self.inputs.append({key: value.clone() for key, value in kwargs.items() if torch.is_tensor(value)})
        prefix = kwargs["input_ids"]
        self.embedded_inputs.append(self.embedding(prefix).detach().clone())
        suffix = torch.tensor([[5, 2, 0]], dtype=torch.long)
        return torch.cat([prefix, suffix], dim=1)

    def __call__(self, input_ids, attention_mask, **kwargs):
        vocab = 8
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab)
        return SimpleNamespace(logits=logits)


def _bare_hf_backend():
    backend = HFDecoderCollectorBackend.__new__(HFDecoderCollectorBackend)
    backend.model = _HFGenerateModel()
    backend.device = torch.device("cpu")
    backend.identity = CollectorIdentity("llava15", "a" * 64, "b" * 64, "c" * 64, "p", (2,), 0)
    return backend


def test_hf_multimodal_backend_preserves_raw_eos_and_prepared_inputs():
    backend = _bare_hf_backend()
    prepared = {
        "inputs": {
            "input_ids": torch.tensor([[7, 6]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.tensor([[[1.0]]]),
        }
    }
    original_ids = prepared["inputs"]["input_ids"].clone()
    actions = backend.generate_actions(
        prepared, num_samples=1, max_new_tokens=3, prefix_ids=(4,), zero_mask_index=0
    )
    assert actions[0].token_ids == (5, 2)
    assert actions[0].stop_reason == "eos"
    assert torch.equal(prepared["inputs"]["input_ids"], original_ids)
    assert backend.model.inputs[0]["input_ids"].tolist() == [[7, 6, 4]]
    assert torch.count_nonzero(backend.model.embedded_inputs[0][0, 2]) == 0
    assert torch.count_nonzero(backend.model.embedded_inputs[0][0, :2]) > 0
    assert torch.equal(backend.model.inputs[0]["pixel_values"], prepared["inputs"]["pixel_values"])


def test_hf_backend_uses_model_generation_eos_when_tokenizer_disagrees():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1))
    tokenizer = SimpleNamespace(eos_token_id=2)
    assert generation_eos_token_ids(model, tokenizer) == (1,)

    backend = _bare_hf_backend()
    backend.identity = CollectorIdentity("instructblip", "a" * 64, "b" * 64, "c" * 64, "p", (1,), 0)
    backend.model.generate = lambda **kwargs: torch.cat(
        [kwargs["input_ids"], torch.tensor([[5, 2, 1]], dtype=torch.long)], dim=1
    )
    prepared = {"inputs": {"input_ids": torch.tensor([[7, 6]]), "attention_mask": torch.ones(1, 2)}}
    action = backend.generate_actions(prepared, num_samples=1, max_new_tokens=3)[0]
    assert action.token_ids == (5, 2, 1)
    assert action.stop_reason == "eos"


def test_hf_group_sampling_disables_default_top_k_to_match_v3_metadata():
    backend = _bare_hf_backend()
    prepared = {
        "inputs": {
            "input_ids": torch.tensor([[7, 6]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    }
    backend.generate_actions(prepared, num_samples=2, max_new_tokens=3)
    kwargs = backend.model.generation_kwargs[0]
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 0


def test_hf_necessity_fails_closed_if_model_bypasses_input_embedding():
    backend = _bare_hf_backend()
    backend.model.generate = lambda **kwargs: torch.cat(
        [kwargs["input_ids"], torch.tensor([[5, 2]], dtype=torch.long)], dim=1
    )
    prepared = {
        "inputs": {
            "input_ids": torch.tensor([[7, 6]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    }
    with pytest.raises(RuntimeError, match="did not pass through the model input embedding"):
        backend.generate_actions(
            prepared, num_samples=1, max_new_tokens=2, prefix_ids=(4,), zero_mask_index=0
        )


def test_hf_action_scoring_has_one_untempered_log_prob_per_raw_action():
    backend = _bare_hf_backend()
    prepared = {"inputs": {"input_ids": torch.tensor([[7, 6]]), "attention_mask": torch.ones(1, 2)}}
    scores = backend.action_log_probs(prepared, [5, 2])
    assert scores.shape == (2,)
    assert torch.allclose(scores, torch.full((2,), -torch.log(torch.tensor(8.0))))


def test_hf_backend_rejects_response_beyond_checkpoint_context():
    backend = _bare_hf_backend()
    prepared = {"inputs": {"input_ids": torch.ones(1, 30, dtype=torch.long), "attention_mask": torch.ones(1, 30)}}
    with pytest.raises(ValueError, match="context limit"):
        backend.generate_actions(prepared, num_samples=1, max_new_tokens=3)
