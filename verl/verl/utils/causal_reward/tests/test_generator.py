import json

import pytest
import torch

from verl.utils.causal_reward.generator import (
    CausalRewardGenerator,
    load_samples,
    save_rewards,
    score_answer,
)
from verl.utils.causal_reward.integration import COPOConfig
from verl.utils.causal_reward.precomputed import load_offline_causal_reward


class FakeBackend:
    """Deterministic backend: ``<answer>`` maps to token 1000."""

    def tokenize(self, text):
        text = (text or "").strip()
        if text == "<answer>":
            return [1000]
        if text == "</answer>":
            return [1001]
        if text == "think <answer> yes </answer>":
            return [10, 11, 1000, 12, 1001]
        tokens = []
        for ch in text:
            if ch == "<":
                tokens.append(1000)
            elif ch == ">":
                tokens.append(1001)
            elif ch == " ":
                continue
            else:
                tokens.append(ord(ch) % 50 + 10)
        return tokens

    def generate(self, sample, prefix_ids=None, max_new_tokens=32, num_samples=1):
        if prefix_ids is None:
            return ["think <answer> yes </answer>" for _ in range(num_samples)]
        # Distinguish sufficiency "with" (odd prefix length) from "without".
        answer = "yes" if len(prefix_ids) % 2 == 0 else "no"
        return [f"<answer> {answer} </answer>" for _ in range(num_samples)]


def make_generator(**overrides):
    cfg = COPOConfig(enable_sufficiency=True, enable_necessity=True, num_suffix_samples=2)
    kwargs = dict(
        backend=FakeBackend(),
        config=cfg,
        max_response_length=16,
        num_rollouts=2,
        max_causal_tokens=3,
    )
    kwargs.update(overrides)
    return CausalRewardGenerator(**kwargs)


def samples():
    return [
        {"sample_id": "a", "image": "i.jpg", "prompt": "q1", "answer": "yes"},
        {"sample_id": "b", "image": "i.jpg", "prompt": "q2", "answer": "yes"},
    ]


def test_generate_shapes_and_ordering():
    store = make_generator().generate(samples())
    assert store["sufficiency"].shape == (4, 16)
    assert store["necessity"].shape == (4, 16)
    assert store["causal_reward"].shape == (4, 16)
    assert store["response_token_ids"].shape == (4, 16)
    assert torch.equal(store["response_lengths"], torch.tensor([5, 5, 5, 5]))
    assert store["sample_ids"] == ["a::0", "a::1", "b::0", "b::1"]


def test_rewards_are_finite_and_bounded():
    store = make_generator().generate(samples())
    assert torch.isfinite(store["causal_reward"]).all()
    assert (store["sufficiency"] >= 0).all()
    assert (store["necessity"] >= -1).all() and (store["necessity"] <= 1).all()


def test_sufficiency_reflects_prefix_answer_change():
    # pos 1 has prefix lengths 2 (with -> "yes") vs 1 (without -> "no").
    store = make_generator(num_rollouts=1).generate(samples())
    assert store["sufficiency"][0, 1] == 1.0
    assert store["sufficiency"][0, 0] == 0.0


def test_aggregated_causal_reward_is_sum_of_components():
    # r_causal = lambda_s * S_suff + lambda_n * S_nec, with lambda_s = lambda_n = 0.35
    # (both components are zero outside reasoning tokens).
    store = make_generator().generate(samples())
    assert torch.allclose(store["causal_reward"], 0.35 * (store["sufficiency"] + store["necessity"]))


def test_save_and_load_round_trip(tmp_path):
    store = make_generator().generate(samples())
    path = tmp_path / "causal_rewards.pt"
    save_rewards(str(path), store)

    loaded = torch.load(path, weights_only=True)
    assert set(loaded) == {
        "sufficiency", "necessity", "causal_reward", "reasoning_mask",
        "response_token_ids", "response_lengths", "sample_ids"
    }
    assert torch.equal(loaded["causal_reward"], store["causal_reward"])

    meta = json.loads((tmp_path / "causal_rewards.pt.json").read_text())
    assert meta["format_version"] == 2
    assert meta["sample_ids"] == store["sample_ids"]


def test_saved_store_usable_by_precomputed_loader(tmp_path):
    store = make_generator().generate(samples())
    path = tmp_path / "causal_rewards.pt"
    save_rewards(str(path), store)

    cfg = COPOConfig(reward_source="precomputed", precomputed_reward_path=str(path))
    response_mask = torch.arange(16).unsqueeze(0) < store["response_lengths"].unsqueeze(1)
    out = load_offline_causal_reward(
        response_mask,
        cfg,
        responses=store["response_token_ids"],
        sample_ids=["a", "a", "b", "b"],
    )
    assert out.shape == (4, 16)


def test_reasoning_only_precomputed_store_without_mask_is_rejected(tmp_path):
    store = make_generator().generate(samples())
    store.pop("reasoning_mask")
    path = tmp_path / "causal_rewards_without_mask.pt"
    torch.save({key: value for key, value in store.items() if key in {
        "sample_ids", "response_token_ids", "response_lengths", "causal_reward"
    }}, path)
    cfg = COPOConfig(reward_source="precomputed", precomputed_reward_path=str(path))
    response_mask = torch.arange(16).unsqueeze(0) < store["response_lengths"].unsqueeze(1)
    with pytest.raises(ValueError, match="explicit reasoning mask"):
        load_offline_causal_reward(
            response_mask,
            cfg,
            responses=store["response_token_ids"],
            sample_ids=["a", "a", "b", "b"],
        )


def test_precomputed_loader_rejects_a_different_response(tmp_path):
    store = make_generator().generate(samples())
    path = tmp_path / "causal_rewards.pt"
    save_rewards(str(path), store)
    cfg = COPOConfig(reward_source="precomputed", precomputed_reward_path=str(path))
    response_mask = torch.arange(16).unsqueeze(0) < store["response_lengths"].unsqueeze(1)
    responses = store["response_token_ids"].clone()
    responses[0, 0] += 1
    with pytest.raises(ValueError, match="trajectory mismatch"):
        load_offline_causal_reward(
            response_mask, cfg, responses=responses, sample_ids=["a", "a", "b", "b"]
        )


def test_precomputed_loader_rejects_unverifiable_legacy_store(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"causal_reward": torch.zeros(1, 4)}, path)
    cfg = COPOConfig(reward_source="precomputed", precomputed_reward_path=str(path))
    with pytest.raises(ValueError, match="no verifiable trajectory identity"):
        load_offline_causal_reward(
            torch.ones(1, 4), cfg, responses=torch.ones(1, 4, dtype=torch.long), sample_ids=["a"]
        )


def test_precomputed_loader_matches_rows_by_sample_and_tokens(tmp_path):
    path = tmp_path / "aligned.pt"
    torch.save(
        {
            "sample_ids": ["a::0", "b::0"],
            "response_token_ids": torch.tensor([[10, 11, 0], [20, 21, 0]]),
            "response_lengths": torch.tensor([2, 2]),
            "causal_reward": torch.tensor([[1.0, 1.5, 0.0], [2.0, 2.5, 0.0]]),
        },
        path,
    )
    cfg = COPOConfig(
        reward_source="precomputed", precomputed_reward_path=str(path), reward_scope="full_sequence"
    )
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    responses = torch.tensor([[20, 21, 0], [10, 11, 0]])
    out = load_offline_causal_reward(
        response_mask, cfg, responses=responses, sample_ids=["b", "a"]
    )
    assert torch.equal(out, torch.tensor([[2.0, 2.5, 0.0], [1.0, 1.5, 0.0]]))


def test_load_samples_requires_fields(tmp_path):
    data = tmp_path / "samples.jsonl"
    data.write_text('{"sample_id": "x", "image": "i.jpg", "prompt": "q"}\n')
    with pytest.raises(ValueError):
        load_samples(str(data))


def test_load_samples_invalid_json(tmp_path):
    data = tmp_path / "samples.jsonl"
    data.write_text("not json\n")
    with pytest.raises(ValueError):
        load_samples(str(data))


def test_load_samples_reads_records(tmp_path):
    data = tmp_path / "samples.jsonl"
    data.write_text('{"sample_id": "a", "image": "i.jpg", "prompt": "q", "answer": "yes"}\n')
    records = load_samples(str(data))
    assert records == [{"sample_id": "a", "image": "i.jpg", "prompt": "q", "answer": "yes"}]


def test_score_answer_binary_match():
    assert score_answer("<answer> yes </answer>", "yes", "strict", "binary") == 1.0


def test_generator_rejects_response_token_truncation():
    class TooLongBackend(FakeBackend):
        def tokenize(self, text):
            if text in {"<answer>", "</answer>"}:
                return super().tokenize(text)
            return list(range(20))

    generator = make_generator(backend=TooLongBackend(), max_response_length=8, num_rollouts=1)
    with pytest.raises(ValueError, match="refusing to truncate"):
        generator.generate(samples()[:1])


def test_reasoning_delimiter_is_matched_as_an_exact_token_sequence():
    generator = make_generator()
    assert generator._reasoning_positions([7, 8, 100, 101, 9], [100, 101]) == [0, 1]
    # Seeing the two delimiter token IDs separately is not a delimiter match.
    assert generator._reasoning_positions([7, 100, 8, 101, 9], [100, 101]) == [0, 1, 2, 3, 4]


def test_necessity_prefix_never_contains_store_width_padding_or_original_answer():
    class RecordingBackend(FakeBackend):
        def __init__(self):
            self.prefixes = []

        def generate(self, sample, prefix_ids=None, max_new_tokens=32, num_samples=1):
            if prefix_ids is not None:
                self.prefixes.append(list(prefix_ids))
            return super().generate(sample, prefix_ids, max_new_tokens, num_samples)

    backend = RecordingBackend()
    make_generator(backend=backend, num_rollouts=1, max_response_length=16).generate(samples()[:1])
    # Response tokens are [reasoning(2), <answer>, answer, </answer>].  All
    # counterfactual prefixes stop at the answer delimiter; none are padded to 16
    # or include the original answer token at index 3.
    assert backend.prefixes
    assert max(map(len, backend.prefixes)) <= 3
