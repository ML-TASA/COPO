"""GRPO group-normalized advantage and policy-gradient loss."""

from __future__ import annotations

import torch


def group_advantage(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Per-group z-normalization.

    ``rewards`` is flattened ``(num_groups * group_size,)``; each group of
    ``group_size`` is normalized independently:
    ``adv_i = (r_i - mean(group)) / (std(group) + eps)``.
    """
    r = rewards.float().reshape(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    return ((r - mean) / (std + eps)).reshape(-1)


def policy_gradient_loss(advantages: torch.Tensor, response_log_probs: torch.Tensor,
                         response_mask: torch.Tensor = None) -> torch.Tensor:
    """``loss = -mean(advantage_i * sum_t log p(response_t | image, instruction))``."""
    masked = response_log_probs if response_mask is None else response_log_probs * response_mask.float()
    per_sample = masked.sum(dim=1)  # (G,)
    return -(advantages * per_sample).mean()
