"""User-facing validation of the COPO causal-reward advantage math."""

import pytest
import torch

from verl.utils.causal_reward.integration import (
    COPOConfig,
    apply_copo_advantage,
    build_causal_reward_tensor,
)
from verl.utils.causal_reward.generation import VerlRolloutGenerationCallable
from verl.utils.causal_reward.reward_manager import (
    build_necessity_intervention,
    build_reasoning_answer_masks,
    sufficiency_score,
)


def mask():
    return torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.float32)


def test_copo_enabled_changes_advantage():
    original = torch.zeros(2, 4)
    assert apply_copo_advantage(original, mask(), torch.ones(2, 4), True).sum() == 6


def test_causal_reward_shape_matches_response_mask():
    with pytest.raises(ValueError):
        build_causal_reward_tensor(mask(), COPOConfig(), causal_reward=torch.ones(2, 3))


def test_reasoning_only_reward_does_not_touch_answer_tokens():
    reasoning = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
    reward = build_causal_reward_tensor(mask(), COPOConfig(), causal_reward=torch.ones(2, 4), reasoning_mask=reasoning)
    assert reward[:, 2:].sum() == 0


def test_reasoning_only_reward_requires_verified_mask():
    with pytest.raises(ValueError, match="explicit reasoning mask"):
        build_causal_reward_tensor(mask(), COPOConfig(), causal_reward=torch.ones(2, 4))


def test_nonfinite_causal_reward_is_rejected():
    reward = torch.ones(2, 4)
    reward[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        build_causal_reward_tensor(mask(), COPOConfig(), causal_reward=reward, reasoning_mask=mask())


def test_full_sequence_reward_touches_response_tokens():
    cfg = COPOConfig(reward_scope="full_sequence")
    assert build_causal_reward_tensor(mask(), cfg, causal_reward=torch.ones(2, 4)).sum() == 6


def test_lambda_s_zero_runs():
    cfg = COPOConfig(lambda_s=0)
    reward = build_causal_reward_tensor(mask(), cfg, sufficiency_reward=torch.ones(2, 4), necessity_reward=torch.ones(2, 4), reasoning_mask=mask())
    assert reward.sum() == pytest.approx(cfg.lambda_n * 6)


def test_lambda_n_zero_runs():
    cfg = COPOConfig(lambda_n=0)
    reward = build_causal_reward_tensor(mask(), cfg, sufficiency_reward=torch.ones(2, 4), necessity_reward=torch.ones(2, 4), reasoning_mask=mask())
    assert reward.sum() == pytest.approx(cfg.lambda_s * 6)


def test_disable_both_suff_nec_falls_back_to_grpo():
    cfg = COPOConfig(enable_sufficiency=False, enable_necessity=False)
    reward = build_causal_reward_tensor(mask(), cfg, sufficiency_reward=torch.ones(2, 4), necessity_reward=torch.ones(2, 4), reasoning_mask=mask())
    assert reward.sum() == 0


def test_padding_positions_zero_reward():
    reward = build_causal_reward_tensor(mask(), COPOConfig(reward_scope="full_sequence"), causal_reward=torch.ones(2, 4))
    assert reward[:, -1].sum() == 0


def test_sufficiency_averages_pairwise_positive_improvements():
    # Eq. 6 applies the indicator to each paired continuation before averaging.
    # max(mean(with) - mean(without), 0) would incorrectly return zero here.
    score = sufficiency_score([1.0, 0.0], [0.0, 1.0])
    assert score == pytest.approx(0.5)


def test_sufficiency_rejects_unpaired_continuations():
    with pytest.raises(ValueError, match="identical shapes"):
        sufficiency_score([1.0, 0.0], [0.0])


def test_multitoken_answer_delimiters_define_exact_reasoning_and_answer_spans():
    responses = torch.tensor([[7, 8, 100, 101, 20, 21, 102, 103, 0]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]])
    reasoning, answer, fallbacks = build_reasoning_answer_masks(
        responses, response_mask, answer_start_ids=(100, 101), answer_end_ids=(102, 103)
    )
    assert fallbacks == 0
    assert torch.equal(reasoning, torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0]]))
    assert torch.equal(answer, torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 0]]))


def test_partial_delimiter_match_disables_causal_mask_instead_of_using_full_response():
    responses = torch.tensor([[7, 100, 8, 101]])
    response_mask = torch.ones_like(responses)
    reasoning, answer, fallbacks = build_reasoning_answer_masks(
        responses, response_mask, answer_start_ids=(100, 101), answer_end_ids=(102, 103)
    )
    assert fallbacks == 1
    assert not reasoning.bool().any()
    assert not answer.bool().any()


def test_missing_answer_end_disables_causal_mask():
    responses = torch.tensor([[7, 8, 100, 101, 20, 21]])
    response_mask = torch.ones_like(responses)
    reasoning, answer, invalid = build_reasoning_answer_masks(
        responses, response_mask, answer_start_ids=(100, 101), answer_end_ids=(102, 103)
    )
    assert invalid == 1
    assert not reasoning.any()
    assert not answer.any()


def test_necessity_intervention_excludes_answer_and_padding():
    response = torch.tensor([7, 8, 100, 101, 20, 21, 102, 103, 0, 0])
    response_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    reasoning = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    masked = build_necessity_intervention(
        response,
        response_mask,
        reasoning,
        token_index=1,
        answer_start_ids=(100, 101),
        mode="zero",
    )
    assert torch.equal(masked, torch.tensor([7, 0, 100, 101]))


def test_necessity_intervention_fallback_still_excludes_padding():
    response = torch.tensor([7, 8, 9, 0, 0])
    response_mask = torch.tensor([1, 1, 1, 0, 0])
    reasoning = response_mask.clone()
    masked = build_necessity_intervention(
        response, response_mask, reasoning, token_index=1, answer_start_ids=(), mode="zero"
    )
    assert torch.equal(masked, torch.tensor([7, 0, 9]))


def test_native_online_necessity_fails_closed_without_embedding_intervention():
    generator = VerlRolloutGenerationCallable(rollout_manager=object())
    with pytest.raises(NotImplementedError, match="embedding zero-mask"):
        generator(mode="necessity", row=0, data=None, masked_response=torch.tensor([7, 0, 9]))
