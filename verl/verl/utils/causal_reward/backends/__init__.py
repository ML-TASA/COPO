"""Model-specific fixed-trajectory collector backends."""

from __future__ import annotations


def build_backend(model_family: str, **kwargs):
    if model_family == "llava15":
        from .llava15 import Llava15CollectorBackend

        return Llava15CollectorBackend(**kwargs)
    if model_family == "qwenvl":
        from .qwen_vl import QwenVLCollectorBackend

        return QwenVLCollectorBackend(**kwargs)
    if model_family == "instructblip":
        from .instructblip import InstructBlipCollectorBackend

        return InstructBlipCollectorBackend(**kwargs)
    if model_family == "minigpt4":
        from .minigpt4 import MiniGPT4CollectorBackend

        return MiniGPT4CollectorBackend(**kwargs)
    raise ValueError(f"unsupported COPO collector model family: {model_family!r}")


__all__ = ["build_backend"]
