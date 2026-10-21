"""Group-relative and token-level causal advantage for GRPO."""

from __future__ import annotations

import torch


def group_advantage(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Per-group z-normalization over flattened ``(B,)`` rewards.

    ``rewards`` is flattened ``(num_groups * group_size,)``; each group of
    ``group_size`` is normalized independently:
    ``adv_i = (r_i - mean(group)) / (std(group) + eps)``.
    """
    r = rewards.float().reshape(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    return ((r - mean) / (std + eps)).reshape(-1)


def causal_advantage_from_group(group_advantage, response_mask, causal_reward):
    """Broadcast the group advantage to token level, add the causal reward, then mask.

    This is the paper's Eq. 9: ``A_hat(i,t) = A_orig(i,t) + r_causal(i,t)``,
    where ``A_orig`` is the group-relative GRPO advantage broadcast over the
    response tokens and ``r_causal`` is the token-level causal reward.
    """
    token_advantage = group_advantage.float().unsqueeze(1).expand_as(response_mask)
    return (token_advantage + causal_reward) * response_mask.float()


def causal_policy_gradient_loss(advantage, response_log_probs):
    """``loss = -mean_i( sum_t advantage_{i,t} * log p(response_t | image, instruction) )``.

    ``advantage`` is already masked per-token, so non-response positions contribute zero.
    """
    return -(advantage * response_log_probs).sum(dim=1).mean()
