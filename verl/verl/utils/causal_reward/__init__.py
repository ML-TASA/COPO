"""COPO causal-reward utilities.

This package implements the COPO causal-completeness reward on top of verl. It
is organized into independent, reusable modules:

- ``integration``: ``COPOConfig`` and the core reward/advantage math.
- ``generation``: the rollout generation callable and ``GenerationResult``.
- ``reward_manager``: online (rollout-time) reward generation.
- ``precomputed``: offline (precomputed) reward loading.
- ``generator``: offline reward generation pipeline.
"""

from verl.utils.causal_reward.integration import (
    COPOConfig,
    apply_copo_advantage,
    build_causal_reward_tensor,
    causal_reward_metrics,
    resolve_copo_enabled,
    validate_copo_config,
)

__all__ = [
    "COPOConfig",
    "apply_copo_advantage",
    "build_causal_reward_tensor",
    "causal_reward_metrics",
    "resolve_copo_enabled",
    "validate_copo_config",
]
