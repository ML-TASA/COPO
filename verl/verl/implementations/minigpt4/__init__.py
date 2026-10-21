"""MiniGPT-4 backend for COPO."""

from __future__ import annotations

__all__ = [
    "MiniGPT4Config",
    "MiniGPT4",
    "load_model",
    "apply_freeze",
    "trainable_parameters",
    "build_optimizer",
    "generate",
    "generate_group",
    "response_log_probs",
    "group_advantage",
    "causal_advantage_from_group",
    "causal_policy_gradient_loss",
    "RewardBackend",
    "OutcomeReward",
    "DummyReward",
    "combine_sufficiency_necessity",
    "OfflineCausalRewardProvider",
    "load_causal_rewards",
    "COPOTrainer",
    "MiniGPT4Processor",
    "MiniGPT4Sample",
    "load_image",
]


def __getattr__(name: str):
    if name == "MiniGPT4Config":
        from .config import MiniGPT4Config

        return MiniGPT4Config
    if name == "MiniGPT4":
        from .mini_gpt4 import MiniGPT4

        return MiniGPT4
    if name in {"load_model", "apply_freeze", "trainable_parameters",
                "build_optimizer", "generate", "generate_group",
                "response_log_probs"}:
        from . import model

        return getattr(model, name)
    if name in {"group_advantage", "causal_advantage_from_group",
                "causal_policy_gradient_loss"}:
        from . import advantage

        return getattr(advantage, name)
    if name in {"RewardBackend", "OutcomeReward", "DummyReward"}:
        from . import reward

        return getattr(reward, name)
    if name in {"combine_sufficiency_necessity", "OfflineCausalRewardProvider",
                "load_causal_rewards"}:
        from . import causal_reward

        return getattr(causal_reward, name)
    if name == "COPOTrainer":
        from .trainer import COPOTrainer

        return COPOTrainer
    if name == "MiniGPT4Processor":
        from .processor import MiniGPT4Processor

        return MiniGPT4Processor
    if name in {"MiniGPT4Sample", "load_image"}:
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
