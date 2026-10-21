from __future__ import annotations

import torch
import pytest

from verl.utils.causal_reward.collector import (
    CollectorIdentity,
    FixedTrajectoryCollector,
    GeneratedAction,
    model_state_fingerprint,
    prefilled_structured_response_masks,
    prefilled_structured_response_masks_from_decode,
    save_fixed_trajectory_store,
    structured_response_masks,
    structured_response_masks_from_decode,
)
from verl.utils.causal_reward.trajectory import FixedTrajectoryStore


class _Backend:
    def __init__(self, policy_id, *, masked_answer=31):
        self.identity = CollectorIdentity(
            model_family="llava15",
            tokenizer_fingerprint="a" * 64,
            processor_template_fingerprint="b" * 64,
            model_config_fingerprint="c" * 64,
            policy_id=policy_id,
            eos_token_ids=(2,),
            pad_token_id=0,
        )
        self.masked_answer = masked_answer
        self.calls = []

    def prepare_sample(self, sample):
        special = {"<think>": [10], "</think>": [11], "<answer>": [12], "</answer>": [13]}
        return {
            "prompt_token_ids": [101, 102],
            "image_hash": "d" * 64,
            "tokenize_special": special.__getitem__,
        }

    def generate_actions(self, prepared, *, num_samples, max_new_tokens, prefix_ids=(), zero_mask_index=None):
        self.calls.append((tuple(prefix_ids), zero_mask_index, num_samples))
        if not prefix_ids:
            ids = (10, 20, 11, 12, 30, 13, 2)
        elif zero_mask_index is not None:
            ids = (11, 12, self.masked_answer, 13, 2)
        elif len(prefix_ids) == 2:
            ids = (11, 12, 30, 13, 2)
        else:
            ids = (11, 12, 31, 13, 2)
        return [GeneratedAction(ids, "eos") for _ in range(num_samples)]

    def action_log_probs(self, prepared, action_ids):
        offset = 0.1 if self.identity.policy_id == "collector" else 0.2
        return -torch.arange(1, len(action_ids) + 1, dtype=torch.float32) * offset

    def decode_action(self, action_ids):
        mapping = {10: "<think>", 20: "reason", 11: "</think>", 12: "<answer>", 30: "yes", 31: "no", 13: "</answer>", 2: ""}
        return " ".join(mapping.get(int(value), "") for value in action_ids)


def _sample():
    return {"sample_id": "s", "group_id": "g", "image_id": "i", "answer": "yes", "task": "binary"}


def test_collector_writes_validator_accepted_v3_and_round_trips(tmp_path):
    collector_backend = _Backend("collector")
    reference = _Backend("reference")
    collector = FixedTrajectoryCollector(
        collector_backend, reference, group_size=2, max_response_length=12, num_suffix_samples=2
    )
    store = collector.collect([_sample()])
    assert store["contract"] == "copo.fixed_trajectory.v3"
    assert store["sample_ids"] == ["s", "s"]
    assert store["group_ids"] == ["g", "g"]
    assert store["metadata"]["sampling_parameters"]["num_suffix_samples"] == 2
    assert torch.equal(store["response_token_ids"][:, :7], torch.tensor([[10, 20, 11, 12, 30, 13, 2]] * 2))
    assert torch.allclose(store["causal_reward"], 0.35 * store["sufficiency"] + 0.35 * store["necessity"])
    assert torch.all(store["sufficiency"][:, 1] == 1)
    assert torch.all(store["necessity"][:, 1] == 1)
    path = tmp_path / "trajectories.pt"
    save_fixed_trajectory_store(path, store)
    loaded = FixedTrajectoryStore.load(path)
    selected = loaded.select_groups(["s"], ["g"], ["i"], ["d" * 64])
    assert selected.rows == (0, 1)


def test_collector_uses_h_paired_with_without_continuations():
    backend = _Backend("collector")
    FixedTrajectoryCollector(
        backend, _Backend("reference"), group_size=2, max_response_length=12, num_suffix_samples=3
    ).collect([_sample()])
    with_calls = [call for call in backend.calls if call[0] == (10, 20) and call[1] is None]
    without_calls = [call for call in backend.calls if call[0] == (10,) and call[1] is None]
    assert with_calls and without_calls
    assert all(call[2] == 3 for call in with_calls + without_calls)


def test_structural_masks_fail_closed_instead_of_treating_full_response_as_reasoning():
    try:
        structured_response_masks(
            [20, 30, 2], think_start_ids=[10], think_end_ids=[11], answer_start_ids=[12], answer_end_ids=[13]
        )
    except ValueError as exc:
        assert "complete <think>/<answer>" in str(exc)
    else:
        raise AssertionError("malformed structured response was accepted")


def test_structural_masks_support_bpe_tokens_crossing_delimiter_boundaries():
    pieces = ["<th", "ink", ">", "reason", ".</", "think", "><", "answer", ">", "no", "</", "answer", ">"]

    def decode(ids):
        return "".join(pieces[int(token)] for token in ids)

    action = tuple(range(len(pieces)))
    reasoning, answer, special = structured_response_masks_from_decode(action, decode)
    assert torch.where(reasoning)[0].tolist() == [3]
    assert torch.where(answer)[0].tolist() == [9]
    assert torch.all(special == ~(reasoning | answer))


def test_prompt_prefilled_think_masks_keep_only_sampled_reasoning_and_answer():
    ids = [20, 21, 11, 12, 30, 13, 2]
    reasoning, answer, special = prefilled_structured_response_masks(
        ids, think_end_ids=[11], answer_start_ids=[12], answer_end_ids=[13]
    )
    assert torch.where(reasoning)[0].tolist() == [0, 1]
    assert torch.where(answer)[0].tolist() == [4]
    assert torch.where(special)[0].tolist() == [2, 3, 5, 6]


def test_prompt_prefilled_think_decode_fallback_preserves_action_ids():
    pieces = ["reason", ".</", "think", "><", "answer", ">", "no", "</", "answer", ">"]

    def decode(ids):
        return "".join(pieces[int(token)] for token in ids)

    action = tuple(range(len(pieces)))
    reasoning, answer, special = prefilled_structured_response_masks_from_decode(action, decode)
    assert torch.where(reasoning)[0].tolist() == [0]
    assert torch.where(answer)[0].tolist() == [6]
    assert torch.all(special == ~(reasoning | answer))


def test_instructblip_sentencepiece_prefilled_protocol_keeps_opening_tag_out_of_action():
    pieces = ["I inspect", " the image", " carefully.</", "think", "><", "answer", ">", "yes", "</", "answer", ">", "</s>"]

    def decode(ids):
        return "".join(pieces[int(token)] for token in ids)

    action = tuple(range(len(pieces)))
    reasoning, answer, special = prefilled_structured_response_masks_from_decode(action, decode)
    assert "<think>" not in decode(action)
    assert torch.where(reasoning)[0].tolist() == [0, 1]
    assert torch.where(answer)[0].tolist() == [7]
    assert torch.where(special)[0].tolist() == [2, 3, 4, 5, 6, 8, 9, 10, 11]


def test_collector_records_prompt_prefill_for_native_exact_prompt_replay():
    backend = _Backend("collector")
    reference = _Backend("reference")
    original_prepare = backend.prepare_sample
    original_reference_prepare = reference.prepare_sample
    backend.prepare_sample = lambda sample: {**original_prepare(sample), "reasoning_prefill": True}
    reference.prepare_sample = lambda sample: {**original_reference_prepare(sample), "reasoning_prefill": True}

    def generated(prepared, *, num_samples, max_new_tokens, prefix_ids=(), zero_mask_index=None):
        if not prefix_ids:
            ids = (20, 11, 12, 30, 13, 2)
        elif zero_mask_index is not None:
            ids = (11, 12, 31, 13, 2)
        elif len(prefix_ids) == 1:
            ids = (11, 12, 30, 13, 2)
        else:
            ids = (11, 12, 31, 13, 2)
        return [GeneratedAction(ids, "eos") for _ in range(num_samples)]

    backend.generate_actions = generated
    store = FixedTrajectoryCollector(
        backend, reference, group_size=2, max_response_length=12, num_suffix_samples=1
    ).collect([_sample()])
    assert store["metadata"]["sampling_parameters"]["prompt_suffix_token_ids"] == [10]
    assert torch.equal(store["response_token_ids"][:, :6], torch.tensor([[20, 11, 12, 30, 13, 2]] * 2))
    assert torch.equal(store["reasoning_mask"][:, :6], torch.tensor([[1, 0, 0, 0, 0, 0]] * 2))


def test_instructblip_optional_protocol_is_recorded_without_changing_original_stores():
    backend = _Backend("collector")
    reference = _Backend("reference")
    original_prepare = backend.prepare_sample
    original_reference_prepare = reference.prepare_sample
    backend.prepare_sample = lambda sample: {
        **original_prepare(sample),
        "protocol_id": "split_context_two_shot",
    }
    reference.prepare_sample = lambda sample: {
        **original_reference_prepare(sample),
        "protocol_id": "split_context_two_shot",
    }

    candidate = FixedTrajectoryCollector(
        backend, reference, group_size=2, max_response_length=12, num_suffix_samples=1
    ).collect([_sample()])
    original = FixedTrajectoryCollector(
        _Backend("collector"), _Backend("reference"),
        group_size=2, max_response_length=12, num_suffix_samples=1,
    ).collect([_sample()])

    assert candidate["metadata"]["sampling_parameters"]["protocol_id"] == (
        "split_context_two_shot"
    )
    assert "protocol_id" not in original["metadata"]["sampling_parameters"]


def test_collector_reports_invalid_action_identity_without_retry_or_repair():
    backend = _Backend("collector")
    backend.generate_actions = lambda *args, **kwargs: [
        GeneratedAction((20, 30, 2), "eos") for _ in range(kwargs["num_samples"])
    ]
    with pytest.raises(ValueError, match="sample_id='s'.*bounded attempts.*does not modify"):
        FixedTrajectoryCollector(
            backend, _Backend("reference"), group_size=2, max_response_length=12
        ).collect([_sample()])


def test_collector_can_retry_invalid_actions_without_repairing_them():
    backend = _Backend("collector")
    original = backend.generate_actions
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [GeneratedAction((20, 30, 2), "eos") for _ in range(kwargs["num_samples"])]
        return original(*args, **kwargs)

    backend.generate_actions = generate
    store = FixedTrajectoryCollector(
        backend,
        _Backend("reference"),
        group_size=2,
        max_response_length=12,
        max_generation_attempts=2,
    ).collect([_sample()])
    assert calls >= 2
    assert store["metadata"]["sampling_parameters"]["max_generation_attempts"] == 2
    assert torch.equal(
        store["response_token_ids"][:, :7],
        torch.tensor([[10, 20, 11, 12, 30, 13, 2]] * 2),
    )


def test_collector_preserves_negative_necessity():
    backend = _Backend("collector", masked_answer=31)
    sample = dict(_sample(), answer="no")
    store = FixedTrajectoryCollector(
        backend, _Backend("reference"), group_size=2, max_response_length=12, num_suffix_samples=1
    ).collect([sample])
    assert torch.all(store["necessity"][:, 1] == -1)


def test_policy_fingerprint_covers_bfloat16_trainable_state():
    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    before = model_state_fingerprint(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    after = model_state_fingerprint(model)
    assert len(before) == 64
    assert before != after
