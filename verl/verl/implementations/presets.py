"""Model presets: the single source of truth for per-model COPO training settings.

Users select a model and supply only paths; every model-specific and engineering
parameter is filled in from the preset below::

    bash scripts/train.sh --model qwenvl --model-path ... --train-data ... --precomputed-reward ...

    python scripts/prepare_reward/generate_rewards.py --model qwenvl --model-path ... \
        --reference-model ... --dataset ... --output ...

The preset holds only *model* differences (backend, GRPO group size, response
length, learning rate, batch/epoch counts).  Algorithm constants — the Eq. 8
causal-reward weights ``lambda_s`` / ``lambda_n`` — live in the COPO algorithm
config (``COPOConfig`` / ``ppo_trainer.yaml`` / the InstructBLIP and MiniGPT-4
backends' defaults), not here, so selecting a model never changes the algorithm.

``PAPER_REPORTED_SETTINGS`` records Appendix G separately from the released
engineering defaults.  It is intentionally not an executable preset: a local
checkpoint's context window and the selected backend must be compatible before
those values can be used.
"""

from __future__ import annotations

from typing import Any

LLAVA15 = "llava15"
QWENVL = "qwenvl"
INSTRUCTBLIP = "instructblip"
MINIGPT4 = "minigpt4"

#: GRPO group size (samples per prompt). Not pinned by the paper; 8 is a standard
#: GRPO group and a reasonable A100 budget for all four models.
_GROUP_SIZE = 8

#: The two verl-native backends run full-sequence 8k reasoning (paper App G). The
#: two BLIP-2-style backends produce short VQA answers, so they get a generous
#: 2048-token *cap* (the model still stops early at EOS) rather than the short
#: 48/64 used previously, and rather than an unnecessary forced 8192.
_VERL_RESPONSE_LENGTH = 8192
_VQA_RESPONSE_LENGTH = 2048


# Appendix G reports these values collectively for the four experimental
# models.  Keep them separate from MODEL_PRESETS: standard public checkpoints
# do not all support an 8192-token response in addition to the prompt, and the
# released InstructBLIP/MiniGPT-4 adapter trainers are not FSDP/vLLM backends.
PAPER_REPORTED_SETTINGS: dict[str, Any] = {
    "hardware": "A100 GPU clusters",
    "fsdp_parameter_offload": True,
    "fsdp_optimizer_offload": True,
    "vllm_gpu_memory_utilization": 0.6,
    "train_batch_size": 512,
    "mini_batch_size": 256,
    "max_prompt_length": 1024,
    "max_response_length": 8192,
    "lr": 1e-6,
    "mixed_precision": True,
    "offline_fixed_trajectories": True,
    "kl_loss_type": "low_var_kl",
}


def _verl_preset(description: str) -> dict[str, Any]:
    return {
        "backend": "verl",
        "description": description,
        "group_size": _GROUP_SIZE,           # engineering
        "max_response_length": _VERL_RESPONSE_LENGTH,  # paper App G
        "max_prompt_length": 1024,           # paper App G
        "lr": 1e-6,                          # paper App G
        "train_batch_size": 512,             # paper App G
        "mini_batch_size": 256,              # paper App G
        "total_epochs": 3,                   # engineering
        "kl_loss_coef": 0.01,                # engineering; App G names low-var KL, not beta
    }


def _adapter_preset(backend: str, description: str) -> dict[str, Any]:
    # The InstructBLIP / MiniGPT-4 backends fine-tune a frozen-backbone adapter, so
    # the higher 1e-5 learning rate is standard; everything else mirrors COPO.
    return {
        "backend": backend,
        "description": description,
        "group_size": _GROUP_SIZE,           # engineering (was 2 / 4)
        "max_response_length": _VQA_RESPONSE_LENGTH,  # engineering (was 64 / 48)
        "lr": 1e-5,                          # engineering (adapter LR)
        "clip_ratio": 0.2,                  # Eq. 11 PPO clipping
        "kl_loss_coef": 0.01,               # engineering; App G names low-var KL, not beta
    }


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    LLAVA15: _verl_preset("LLaVA-1.5-7B"),
    QWENVL: _verl_preset("Qwen-VL-Chat"),
    INSTRUCTBLIP: _adapter_preset("instructblip", "InstructBLIP"),
    MINIGPT4: _adapter_preset("minigpt4", "MiniGPT-4"),
}


def model_names() -> tuple[str, ...]:
    return tuple(MODEL_PRESETS)


def get_preset(name: str) -> dict[str, Any]:
    """Return a copy of the preset for ``name``, raising on an unknown model."""
    try:
        return dict(MODEL_PRESETS[name])
    except KeyError:
        choices = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"unknown model {name!r}; choose one of: {choices}") from None


def get_paper_reported_settings() -> dict[str, Any]:
    """Return Appendix G's shared settings without applying them to a model."""
    return dict(PAPER_REPORTED_SETTINGS)
