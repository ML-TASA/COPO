from dataclasses import dataclass

import torch


@dataclass
class COPOConfig:
    enable: bool = False
    lambda_s: float = 0.35
    lambda_n: float = 0.35
    enable_sufficiency: bool = True
    enable_necessity: bool = True
    num_suffix_samples: int = 8
    reward_scope: str = "reasoning_only"
    masking: str = "zero"
    advantage_mode: str = "additive"
    reward_source: str = "auto"
    precomputed_reward_path: str = ""
    generation_callable: str = ""
    generation_model: str = ""
    max_causal_tokens: int | None = None
    answer_start_token_ids: tuple = ()
    max_reasoning_tokens_for_copo: int | None = None
    max_generation_failures: int = 0
    generation_timeout: float = 120.0
    answer_normalization: str = "strict"
    answer_end_token_ids: tuple = ()
    reward_cache: object = None
    eps: float = 1e-6


def resolve_copo_enabled(name, enable=False):
    if name not in {"grpo", "copo"}:
        raise ValueError(f"algorithm.name must be grpo or copo, got {name}")
    return name == "copo"


def validate_copo_config(name, cfg):
    if cfg.reward_source not in {"precomputed", "online", "auto"}:
        raise ValueError(f"invalid copo.reward_source: {cfg.reward_source}")
    enabled = resolve_copo_enabled(name, cfg.enable)
    if cfg.reward_scope not in {"reasoning_only", "full_sequence"}:
        raise ValueError(f"invalid copo.reward_scope: {cfg.reward_scope}")
    if cfg.masking not in {"zero", "mean", "noise"}:
        raise ValueError(f"invalid copo.masking: {cfg.masking}")
    if cfg.advantage_mode != "additive":
        raise ValueError(f"invalid copo.advantage_mode: {cfg.advantage_mode}")
    if cfg.num_suffix_samples < 1 or cfg.eps <= 0:
        raise ValueError("copo sample count and eps must be positive")
    if cfg.lambda_s < 0 or cfg.lambda_n < 0:
        raise ValueError("copo lambdas must be non-negative")
    return enabled


def build_causal_reward_tensor(response_mask, cfg, causal_reward=None, sufficiency_reward=None, necessity_reward=None, reasoning_mask=None, special_token_mask=None, image_token_mask=None):
    shape = response_mask.shape
    tensors = [x for x in (causal_reward, sufficiency_reward, necessity_reward) if x is not None]
    if any(x.shape != shape for x in tensors):
        raise ValueError(f"causal reward shape must match response mask shape {tuple(shape)}")
    if causal_reward is None:
        result = torch.zeros_like(response_mask, dtype=torch.float32)
        if cfg.enable_sufficiency and sufficiency_reward is not None:
            result = result + cfg.lambda_s * sufficiency_reward
        if cfg.enable_necessity and necessity_reward is not None:
            result = result + cfg.lambda_n * necessity_reward
    else:
        result = causal_reward.float()
    if cfg.reward_scope == "reasoning_only":
        if reasoning_mask is None:
            raise ValueError("reasoning_only COPO reward requires an explicit reasoning mask")
        if reasoning_mask.shape != shape:
            raise ValueError(f"reasoning mask shape must match response mask shape {tuple(shape)}")
        allowed = reasoning_mask
    else:
        allowed = response_mask
    allowed = allowed * response_mask
    if special_token_mask is not None:
        allowed = allowed * (1 - special_token_mask)
    if image_token_mask is not None:
        allowed = allowed * (1 - image_token_mask)
    result = result * allowed
    if not torch.isfinite(result).all():
        raise ValueError("COPO causal reward contains NaN or inf")
    return result


def apply_copo_advantage(original, response_mask, causal_reward, enabled):
    if not enabled:
        return original
    if causal_reward.shape != response_mask.shape or original.shape != response_mask.shape:
        raise ValueError("advantage, causal reward, and response mask must have identical shapes")
    return (original + causal_reward) * response_mask


def causal_reward_metrics(reward, response_mask):
    valid = reward[response_mask.bool()]
    values = [0.0] * 5 if valid.numel() == 0 else [valid.mean().item(), valid.std(unbiased=False).item(), valid.min().item(), valid.max().item(), (valid != 0).float().mean().item()]
    return dict(zip(("copo/reward_mean", "copo/reward_std", "copo/reward_min", "copo/reward_max", "copo/reward_nonzero_ratio"), values, strict=True))
