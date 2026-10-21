"""Causal advantage integration (offline)."""

from __future__ import annotations

import torch


def apply_causal_advantage(original, response_mask, causal_reward, enabled=True):
    """``(original + causal_reward) * response_mask`` (identical semantics to verl)."""
    if not enabled:
        return original
    if causal_reward.shape != response_mask.shape or original.shape != response_mask.shape:
        raise ValueError("advantage, causal reward and response mask must have identical shapes")
    return (original + causal_reward) * response_mask


def causal_advantage_from_group(group_advantage, response_mask, causal_reward, enabled=True):
    """Broadcast the per-sample group advantage to token level, add the causal reward, then mask."""
    token_advantage = group_advantage.float().unsqueeze(1).expand_as(response_mask)
    return apply_causal_advantage(token_advantage, response_mask, causal_reward, enabled)


def causal_policy_gradient_loss(causal_advantage, response_log_probs):
    """``loss = -mean_i( sum_t advantage_{i,t} * log p(response_t | image, instruction) )``.

    ``causal_advantage`` is already masked per-token, so non-response positions
    contribute zero.
    """
    return -(causal_advantage * response_log_probs).sum(dim=1).mean()
