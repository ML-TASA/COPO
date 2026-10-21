"""COPO integration for the InstructBLIP backend."""

from __future__ import annotations

from .causal_advantage import apply_causal_advantage, causal_advantage_from_group, causal_policy_gradient_loss
from .causal_reward import (
    CausalRewardConfig,
    CausalRewardProvider,
    OfflineCausalRewardProvider,
    build_causal_reward_tensor,
    save_precomputed_causal_rewards,
)

__all__ = [
    "CausalRewardConfig",
    "CausalRewardProvider",
    "OfflineCausalRewardProvider",
    "build_causal_reward_tensor",
    "save_precomputed_causal_rewards",
    "apply_causal_advantage",
    "causal_advantage_from_group",
    "causal_policy_gradient_loss",
]
