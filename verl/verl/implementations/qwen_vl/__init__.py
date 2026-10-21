"""Qwen-VL-Chat (2023) compatibility for the verl COPO pipeline."""

from __future__ import annotations

from typing import Any


def is_qwen_vl(config: Any) -> bool:
    """Return True if ``config`` is the Qwen-VL-Chat (2023) checkpoint."""
    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None)
    return model_type == "qwen" and architectures == ["QWenLMHeadModel"]


__all__ = [
    "is_qwen_vl",
    "install_qwen_vl_compat",
    "QWenVLProcessor",
    "wrap_qwen_vl_for_verl",
    "dedup_qwen_vl_image_tokens",
    "resolve_image_path",
    "ensure_qwen_vl_tokenizer",
]


def __getattr__(name: str):
    if name == "install_qwen_vl_compat":
        from .legacy_compat import install_qwen_vl_compat

        return install_qwen_vl_compat
    if name == "QWenVLProcessor":
        from .processor import QWenVLProcessor

        return QWenVLProcessor
    if name == "wrap_qwen_vl_for_verl":
        from .model_wrapper import wrap_qwen_vl_for_verl

        return wrap_qwen_vl_for_verl
    if name == "dedup_qwen_vl_image_tokens":
        from .input_adapter import dedup_qwen_vl_image_tokens

        return dedup_qwen_vl_image_tokens
    if name == "resolve_image_path":
        from .input_adapter import resolve_image_path

        return resolve_image_path
    if name == "ensure_qwen_vl_tokenizer":
        from .processor import ensure_qwen_vl_tokenizer

        return ensure_qwen_vl_tokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
