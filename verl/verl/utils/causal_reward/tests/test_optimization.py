import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_policy_loss_copo_eq11,
    compute_policy_loss_vanilla,
    kl_penalty,
)
from verl.utils.causal_reward.optimization import (
    compute_copo_eq11_policy_loss,
    compute_copo_eq11_total_loss,
    compute_low_variance_reference_kl,
)


def test_ratio_one_matches_hand_computed_per_trajectory_mean():
    old = torch.zeros(2, 3)
    current = torch.zeros(2, 3, requires_grad=True)
    advantages = torch.tensor([[1.0, 3.0, 100.0], [-2.0, 100.0, 100.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    loss, metrics = compute_copo_eq11_policy_loss(old, current, advantages, mask, 0.2, 0.2)
    assert loss.item() == pytest.approx(0.0)  # -mean(mean(1,3), mean(-2))
    assert metrics["actor/ppo_kl"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_loss"),
    [
        (2.0, 1.0, -1.2),
        (0.5, -1.0, 0.8),
        (0.5, 1.0, -0.5),
        (2.0, -1.0, 2.0),
    ],
)
def test_standard_upper_lower_clip_for_positive_and_negative_advantage(ratio, advantage, expected_loss):
    old = torch.zeros(1, 1)
    current = torch.tensor([[np.log(ratio)]], requires_grad=True)
    loss, _ = compute_copo_eq11_policy_loss(
        old, current, torch.tensor([[advantage]]), torch.ones(1, 1), 0.2, 0.2
    )
    assert loss.item() == pytest.approx(expected_loss)


def test_variable_lengths_use_sequence_then_batch_normalization():
    old = torch.zeros(2, 3)
    current = torch.zeros(2, 3)
    advantages = torch.tensor([[1.0, 3.0, 99.0], [10.0, 10.0, 10.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    loss, _ = compute_copo_eq11_policy_loss(old, current, advantages, mask, 0.2, 0.2)
    assert loss.item() == pytest.approx(-6.0)  # -mean(2, 10), not token mean -6.8


def test_eq11_is_not_verl_dual_clip_for_extreme_negative_advantage():
    old = torch.zeros(1, 1)
    current = torch.tensor([[4.0]])
    advantage = torch.tensor([[-1.0]])
    mask = torch.ones(1, 1)
    strict, _ = compute_copo_eq11_policy_loss(old, current, advantage, mask, 0.2, 0.2)
    generic_config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "global_batch_info": {},
        }
    )
    dual, _ = compute_policy_loss_vanilla(old, current, advantage, mask, config=generic_config)
    assert strict.item() == pytest.approx(np.exp(4.0), rel=1e-5)
    assert dual.item() == pytest.approx(3.0)


def test_registered_eq11_mode_rejects_wrong_aggregation_and_rollout_weights():
    config = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2})
    values = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="seq-mean-token-mean"):
        compute_policy_loss_copo_eq11(
            values, values, values, torch.ones_like(values), loss_agg_mode="token-mean", config=config
        )
    with pytest.raises(ValueError, match="rollout-correction"):
        compute_policy_loss_copo_eq11(
            values,
            values,
            values,
            torch.ones_like(values),
            loss_agg_mode="seq-mean-token-mean",
            config=config,
            rollout_is_weights=torch.ones_like(values),
        )


def test_low_variance_reference_kl_matches_formula_and_masks_padding():
    current = torch.tensor([[-1.0, -2.0, 50.0]], requires_grad=True)
    reference = torch.tensor([[-1.2, -1.8, -50.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    value = kl_penalty(current, reference, "low_var_kl")
    delta = (reference[:, :2] - current[:, :2]).clamp(-20, 20)
    expected = torch.exp(delta) - delta - 1
    assert torch.allclose(value[:, :2], expected)
    masked_mean = (value * mask).sum() / mask.sum()
    masked_mean.backward()
    assert torch.isfinite(current.grad[:, :2]).all()
    assert current.grad[0, 2] == 0
    assert reference.grad is None


def test_shared_adapter_total_loss_matches_native_policy_plus_kl():
    old = torch.tensor([[-1.1, -1.2, 0.0], [-1.0, -1.3, -1.4]])
    current = torch.tensor([[-1.0, -1.4, 7.0], [-1.2, -1.1, -1.5]], requires_grad=True)
    reference = torch.tensor([[-1.3, -1.0, 0.0], [-1.4, -1.2, -1.1]])
    advantage = torch.tensor([[1.0, -1.0, 99.0], [-0.5, 0.5, 1.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    policy, _ = compute_copo_eq11_policy_loss(old, current, advantage, mask, 0.2, 0.2)
    kl = kl_penalty(current, reference, "low_var_kl")
    lengths = mask.sum(dim=-1)
    native_kl = ((kl * mask).sum(dim=-1) / lengths).mean()
    total, metrics = compute_copo_eq11_total_loss(
        old, current, reference, advantage, mask, 0.2, 0.2, 0.01
    )
    assert torch.allclose(total, policy + 0.01 * native_kl)
    assert metrics["actor/kl_loss"] == pytest.approx(native_kl.item())
    total.backward()
    assert torch.isfinite(current.grad[mask.bool()]).all()
    assert torch.count_nonzero(current.grad[~mask.bool()]) == 0


def test_shared_adapter_total_loss_rejects_trainable_old_or_reference():
    current = torch.zeros(1, 1, requires_grad=True)
    mask = torch.ones(1, 1)
    with pytest.raises(ValueError, match="old-policy"):
        compute_copo_eq11_total_loss(
            torch.zeros(1, 1, requires_grad=True), current, torch.zeros(1, 1), mask, mask, 0.2, 0.2, 0.01
        )
    with pytest.raises(ValueError, match="reference"):
        compute_low_variance_reference_kl(current, torch.zeros(1, 1, requires_grad=True), mask)


def test_zero_variance_group_is_finite_and_zero():
    rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    mask = torch.ones_like(rewards)
    advantage, _ = compute_grpo_outcome_advantage(
        rewards, mask, np.array(["group", "group"], dtype=object), norm_adv_by_std_in_grpo=True
    )
    assert torch.isfinite(advantage).all()
    assert torch.count_nonzero(advantage) == 0


def test_eq11_has_finite_nonzero_gradients_only_on_valid_actions():
    old = torch.zeros(2, 3)
    current = torch.tensor([[0.0, 0.1, 50.0], [-0.1, 0.0, -50.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, -1.0, 9.0], [-1.0, 1.0, 9.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    loss, _ = compute_copo_eq11_policy_loss(old, current, advantages, mask, 0.2, 0.2)
    loss.backward()
    assert torch.isfinite(current.grad[mask.bool()]).all()
    assert torch.count_nonzero(current.grad[mask == 0]) == 0
    assert torch.count_nonzero(current.grad[mask.bool()]) > 0
