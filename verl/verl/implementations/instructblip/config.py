"""Configuration for the InstructBLIP backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InstructBlipConfig:
    # --- model loading ---
    model_name_or_path: str = ""

    # --- architecture ---
    num_query_token: int = 32
    image_size: int = 224

    # --- length limits ---
    max_txt_len: int = 2048      # language-model response length (cap; model stops at EOS)
    qformer_max_len: int = 128   # Q-Former instruction length

    # --- generation protocol ---
    # ``original`` is the paper/legacy prompt-prefill path.  The split-context
    # option is an explicit no-training adaptation for checkpoints that follow
    # the output format more reliably when Q-Former sees only the visual
    # question and the language model sees two format-only demonstrations.
    generation_protocol: str = "original"

    # --- freeze policy ---
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = True
    freeze_qformer: bool = False
    freeze_projection: bool = False

    # --- training ---
    lr: float = 1e-5
    weight_decay: float = 0.05
    batch_size: int = 8  # GRPO group size (responses sampled per prompt)
    num_steps: int = 3
    clip_ratio: float = 0.2
    kl_loss_coef: float = 0.01

    # --- dtype / device ---
    dtype: str = "float32"   # one of float16 / bfloat16 / float32
    device: str = "auto"     # "auto" -> cuda if available else cpu

    def to_torch_dtype(self) -> torch.dtype:
        try:
            return {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.dtype]
        except KeyError as exc:
            raise ValueError(f"invalid dtype: {self.dtype!r}") from exc

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def validate(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("InstructBlipConfig.model_name_or_path must be set")
        if self.num_query_token < 1:
            raise ValueError("num_query_token must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size/group_size must be at least 2 for GRPO")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if not 0 <= self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in [0, 1)")
        if self.kl_loss_coef < 0:
            raise ValueError("kl_loss_coef must be non-negative")
        if self.generation_protocol not in {"original", "split_context_two_shot"}:
            raise ValueError(
                "generation_protocol must be 'original' or 'split_context_two_shot'"
            )
        self.to_torch_dtype()  # raises on an invalid dtype
