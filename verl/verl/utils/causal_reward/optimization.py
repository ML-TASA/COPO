"""Policy optimization primitives specific to the COPO Eq. 11 objective."""

from __future__ import annotations

from typing import Any

import torch


def _validate_eq11_inputs(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    shape = response_mask.shape
    if response_mask.ndim != 2:
        raise ValueError(f"COPO Eq. 11 expects 2D [batch, response] tensors, got {tuple(shape)}")
    for name, tensor in (
        ("old_log_prob", old_log_prob),
        ("log_prob", log_prob),
        ("advantages", advantages),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != response_mask shape {tuple(shape)}")
    mask = response_mask.bool()
    lengths = mask.sum(dim=-1)
    if torch.any(lengths == 0):
        rows = torch.where(lengths == 0)[0].tolist()
        raise ValueError(f"COPO Eq. 11 received trajectories with no valid actions at rows {rows}")
    for name, tensor in (
        ("old_log_prob", old_log_prob),
        ("log_prob", log_prob),
        ("advantages", advantages),
    ):
        if not torch.isfinite(tensor[mask]).all():
            raise ValueError(f"{name} contains non-finite values on valid response tokens")
    return mask


def compute_copo_eq11_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy term in COPO Eq. 11.

    The objective averages tokens within each trajectory before averaging the
    trajectories. This implements ``1/G * sum_i 1/T_i * sum_t`` for equal-size
    groups, without the negative-advantage dual-clip extension used by verl's
    generic PPO loss.
    """
    if not 0 <= clip_ratio_low < 1 or clip_ratio_high < 0:
        raise ValueError("COPO Eq. 11 requires 0 <= lower clip < 1 and upper clip >= 0")
    mask = _validate_eq11_inputs(old_log_prob, log_prob, advantages, response_mask)

    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio[mask]).all():
        raise FloatingPointError("COPO Eq. 11 policy ratio is non-finite on valid response tokens")

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    objective = torch.minimum(unclipped, clipped)

    mask_float = mask.to(objective.dtype)
    per_trajectory = (objective * mask_float).sum(dim=-1) / mask_float.sum(dim=-1)
    loss = -per_trajectory.mean()

    clipped_selected = (clipped < unclipped) & mask
    approx_kl = ((old_log_prob - log_prob) * mask_float).sum(dim=-1) / mask_float.sum(dim=-1)
    metrics = {
        "actor/pg_clipfrac": clipped_selected.float().sum().detach().item() / mask.sum().detach().item(),
        "actor/ppo_kl": approx_kl.mean().detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
    return loss, metrics


def compute_low_variance_reference_kl(
    log_prob: torch.Tensor,
    reference_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return native verl's low-variance KL with Eq. 11 normalization."""
    if reference_log_prob.requires_grad:
        raise ValueError("fixed reference log-probabilities must not require gradients")
    mask = _validate_eq11_inputs(reference_log_prob, log_prob, log_prob, response_mask)
    delta = torch.clamp(reference_log_prob - log_prob, min=-20, max=20)
    token_kl = torch.clamp(torch.exp(delta) - delta - 1, min=-10, max=10)
    mask_float = mask.to(token_kl.dtype)
    return ((token_kl * mask_float).sum(dim=-1) / mask_float.sum(dim=-1)).mean()


def compute_copo_eq11_total_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    reference_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    kl_coef: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Eq. 11 policy term plus the reference-policy KL used by native actors.

    Native verl actors call :func:`compute_copo_eq11_policy_loss` and add KL in
    ``dp_actor``. Adapter backends use this wrapper so the KL is added exactly
    once with the same low-variance estimator and sequence-first reduction.
    """
    if old_log_prob.requires_grad:
        raise ValueError("fixed old-policy log-probabilities must not require gradients")
    if kl_coef < 0:
        raise ValueError("KL coefficient must be non-negative")
    policy_loss, metrics = compute_copo_eq11_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        clip_ratio_low,
        clip_ratio_high,
    )
    kl_loss = compute_low_variance_reference_kl(log_prob, reference_log_prob, response_mask)
    total_loss = policy_loss + float(kl_coef) * kl_loss
    metrics = dict(metrics)
    metrics.update(
        {
            "actor/pg_loss": policy_loss.detach().item(),
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/kl_coef": float(kl_coef),
        }
    )
    return total_loss, metrics
