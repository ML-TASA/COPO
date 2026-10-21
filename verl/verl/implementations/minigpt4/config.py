"""Configuration for the MiniGPT-4 backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MiniGPT4Config:
    # --- model loading (official MiniGPT-4 weight sources) ---
    vit_model: str = "eva_clip_g"
    vit_pretrained_path: str = ""   # BLIP-2 eva_vit_g.pth (empty -> random init)
    q_former_model: str = ""        # BLIP-2 blip2_pretrained_flant5xxl.pth (empty -> random init)
    llama_model: str = ""           # Vicuna / LLaMA HF checkpoint directory
    ckpt: str = ""                  # stage-2 projection matching the selected 7B/13B LLM

    # --- architecture (mirrors minigpt4_vicuna0.yaml) ---
    image_size: int = 224
    num_query_token: int = 32
    drop_path_rate: float = 0.0
    use_grad_checkpoint: bool = False
    vit_precision: str = "fp16"
    has_qformer: bool = True

    # --- freeze policy (official: freeze vision + Q-Former + LLM, train projection) ---
    freeze_vit: bool = True
    freeze_qformer: bool = True
    freeze_language_model: bool = True
    freeze_projection: bool = False

    # --- prompt / generation ---
    prompt_template: str = "###Human: {} ###Assistant: "
    end_sym: str = "###"
    max_txt_len: int = 2048  # prompt/response length cap (>= grpo_max_new_tokens)
    max_new_tokens: int = 300
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    use_nucleus_sampling: bool = False

    # --- BERT path for Q-Former config / tokenizer (bert-base-uncased) ---
    # Local snapshot directory (contains config.json + vocab.txt); loaded
    # offline-only. Empty -> fall back to the "bert-base-uncased" hub name.
    bert_model_path: str = ""

    # --- dtype / device ---
    dtype: str = "float16"   # LLM load dtype (official MiniGPT-4 uses fp16)
    device: str = "auto"

    # --- training ---
    lr: float = 1e-5
    weight_decay: float = 0.05

    # --- GRPO ---
    group_size: int = 8          # responses sampled per prompt (must be > 1)
    grpo_steps: int = 2          # rollout+update iterations
    grpo_max_new_tokens: int = 2048  # response length during GRPO rollout
    clip_ratio: float = 0.2
    kl_loss_coef: float = 0.01

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
        required_paths = {
            "llama_model": self.llama_model,
            "vit_pretrained_path": self.vit_pretrained_path,
            "q_former_model": self.q_former_model,
            "ckpt": self.ckpt,
            "bert_model_path": self.bert_model_path,
        }
        for name, value in required_paths.items():
            if not value:
                raise ValueError(f"MiniGPT4Config.{name} must be set for a real MiniGPT-4 run")
            if not Path(value).exists():
                raise FileNotFoundError(f"MiniGPT4Config.{name} path does not exist: {value}")
        if self.num_query_token < 1:
            raise ValueError("num_query_token must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if self.grpo_steps < 1:
            raise ValueError("grpo_steps must be positive")
        if not 0 <= self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in [0, 1)")
        if self.kl_loss_coef < 0:
            raise ValueError("kl_loss_coef must be non-negative")
        self.to_torch_dtype()  # raises on an invalid dtype

    def as_model_kwargs(self) -> dict:
        """The subset of fields consumed by ``MiniGPT4.__init__`` / ``from_config``."""
        return {
            "vit_model": self.vit_model,
            "q_former_model": self.q_former_model,
            "image_size": self.image_size,
            "drop_path_rate": self.drop_path_rate,
            "use_grad_checkpoint": self.use_grad_checkpoint,
            "vit_precision": self.vit_precision,
            "freeze_vit": self.freeze_vit,
            "has_qformer": self.has_qformer,
            "freeze_qformer": self.freeze_qformer,
            "num_query_token": self.num_query_token,
            "llama_model": self.llama_model,
            "llama_dtype": self.dtype,
            "prompt_path": "",
            "prompt_template": self.prompt_template,
            "max_txt_len": self.max_txt_len,
            "end_sym": self.end_sym,
            "bert_model_path": self.bert_model_path or None,
            "vit_pretrained_path": self.vit_pretrained_path or None,
            # stage-2 projection checkpoint (pretrained_minigpt4.pth); consumed by
            # ``MiniGPT4.from_config`` after __init__ builds the model.
            "ckpt": self.ckpt or "",
        }
