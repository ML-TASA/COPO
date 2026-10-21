"""Hugging Face InstructBLIP backend for COPO."""

from __future__ import annotations

__all__ = [
    "InstructBlipConfig",
    "load_model",
    "apply_freeze",
    "build_optimizer",
    "forward_loss",
    "response_log_probs",
    "extract_response_log_probs",
    "generate",
    "generate_responses",
    "group_advantage",
    "policy_gradient_loss",
    "RewardBackend",
    "OutcomeReward",
    "DummyReward",
    "CausalRewardConfig",
    "CausalRewardProvider",
    "OfflineCausalRewardProvider",
    "build_causal_reward_tensor",
    "apply_causal_advantage",
    "causal_advantage_from_group",
    "causal_policy_gradient_loss",
    "InstructBlipProcessor",
    "InstructBlipDataset",
    "InstructBlipSample",
]


def __getattr__(name: str):
    if name == "InstructBlipConfig":
        from .config import InstructBlipConfig

        return InstructBlipConfig
    if name in {"load_model", "apply_freeze", "build_optimizer", "forward_loss",
                "response_log_probs", "extract_response_log_probs", "generate"}:
        from . import model

        return getattr(model, name)
    if name == "generate_responses":
        from .rollout import generate_responses

        return generate_responses
    if name in {"group_advantage", "policy_gradient_loss"}:
        from . import grpo

        return getattr(grpo, name)
    if name in {"RewardBackend", "OutcomeReward", "DummyReward"}:
        from . import reward

        return getattr(reward, name)
    if name in {"CausalRewardConfig", "CausalRewardProvider", "OfflineCausalRewardProvider",
                "build_causal_reward_tensor", "apply_causal_advantage",
                "causal_advantage_from_group", "causal_policy_gradient_loss"}:
        from . import causal

        return getattr(causal, name)
    if name == "InstructBlipProcessor":
        from .processor import InstructBlipProcessor

        return InstructBlipProcessor
    if name in {"InstructBlipDataset", "InstructBlipSample"}:
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
