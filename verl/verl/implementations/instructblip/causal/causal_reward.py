"""Causal reward interface + offline / precomputed sufficiency / necessity loading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from verl.utils.causal_reward.precomputed import match_precomputed_trajectory_rows


@dataclass
class CausalRewardConfig:
    """COPO causal-reward settings (offline subset of verl's COPOConfig)."""

    enable: bool = True
    lambda_s: float = 0.35
    lambda_n: float = 0.35
    enable_sufficiency: bool = True
    enable_necessity: bool = True
    reward_scope: str = "full_sequence"  # "full_sequence" | "reasoning_only"
    advantage_mode: str = "additive"     # only additive supported (matches verl)
    eps: float = 1e-6

    def validate(self) -> None:
        if self.reward_scope not in {"full_sequence", "reasoning_only"}:
            raise ValueError(f"invalid reward_scope: {self.reward_scope!r}")
        if self.advantage_mode != "additive":
            raise ValueError(f"invalid advantage_mode: {self.advantage_mode!r} (only additive)")
        if self.lambda_s < 0 or self.lambda_n < 0:
            raise ValueError("lambda_s / lambda_n must be non-negative")


def build_causal_reward_tensor(
    response_mask,
    config,
    causal_reward=None,
    sufficiency_reward=None,
    necessity_reward=None,
    reasoning_mask=None,
):
    """Combine sufficiency / necessity into a per-token causal reward, masked to the response.

    Mirrors verl's ``build_causal_reward_tensor`` semantics: ``lambda_s *
    sufficiency + lambda_n * necessity`` (or the precomputed ``causal_reward``),
    restricted to ``reward_scope`` and to ``response_mask``. Invalid masks and
    non-finite rewards are rejected, matching the shared formal COPO path.
    """
    shape = response_mask.shape
    tensors = [t for t in (causal_reward, sufficiency_reward, necessity_reward) if t is not None]
    for t in tensors:
        if t.shape != shape:
            raise ValueError(
                f"causal reward shape {tuple(t.shape)} != response mask shape {tuple(shape)}. "
                f"The offline reward store must be response-only [B, L] (L = response length) "
                f"and match the response mask exactly; image / instruction / full-sequence "
                f"positions are not part of the store. Regenerate the store with the same "
                f"rollout count (group_size) and response length as training."
            )
    if causal_reward is not None:
        result = causal_reward.float()
    else:
        result = torch.zeros_like(response_mask, dtype=torch.float32)
        if config.enable_sufficiency and sufficiency_reward is not None:
            result = result + config.lambda_s * sufficiency_reward
        if config.enable_necessity and necessity_reward is not None:
            result = result + config.lambda_n * necessity_reward
    if config.reward_scope == "reasoning_only":
        if reasoning_mask is None:
            raise ValueError("reasoning_only COPO reward requires an explicit reasoning mask")
        if reasoning_mask.shape != shape:
            raise ValueError(f"reasoning mask shape must match response mask shape {tuple(shape)}")
        allowed = reasoning_mask
    else:
        allowed = response_mask
    allowed = allowed * response_mask
    result = result * allowed
    if not torch.isfinite(result).all():
        raise ValueError("COPO causal reward contains NaN or inf")
    return result


class CausalRewardProvider(ABC):
    """Interface: produce a per-token causal reward tensor aligned to ``response_mask``."""

    @abstractmethod
    def get_causal_reward(self, response_mask, reasoning_mask=None, response_token_ids=None,
                          sample_ids=None) -> torch.Tensor:
        ...


class OfflineCausalRewardProvider(CausalRewardProvider):
    """Loads precomputed sufficiency / necessity tensors (from a file or in-memory).

    ``path`` (optional) points to a ``torch.save``'d dict ``{"sufficiency", "necessity"}``.
    In-memory ``sufficiency`` / ``necessity`` are used when no file is given.
    Shapes are validated against ``response_mask``.
    """

    def __init__(self, config, path=None, sufficiency=None, necessity=None):
        self.config = config
        self.path = path
        self.sufficiency = sufficiency
        self.necessity = necessity

    def get_causal_reward(self, response_mask, reasoning_mask=None, response_token_ids=None,
                          sample_ids=None):
        sufficiency, necessity = self.sufficiency, self.necessity
        causal_reward = None
        if self.path is not None:
            data = torch.load(self.path, map_location=response_mask.device, weights_only=True)
            rows = match_precomputed_trajectory_rows(
                data, response_token_ids, response_mask, sample_ids
            )
            causal_reward = data.get("causal_reward")
            sufficiency = data.get("sufficiency", sufficiency)
            necessity = data.get("necessity", necessity)
            if causal_reward is not None:
                causal_reward = causal_reward[rows, : response_mask.shape[1]]
            if sufficiency is not None:
                sufficiency = sufficiency[rows, : response_mask.shape[1]]
            if necessity is not None:
                necessity = necessity[rows, : response_mask.shape[1]]
        return build_causal_reward_tensor(
            response_mask,
            self.config,
            causal_reward=causal_reward,
            sufficiency_reward=sufficiency,
            necessity_reward=necessity,
            reasoning_mask=reasoning_mask,
        )


def save_precomputed_causal_rewards(path, sufficiency, necessity) -> None:
    """Write precomputed sufficiency / necessity tensors to ``path`` (serialization helper)."""
    torch.save(
        {"sufficiency": sufficiency.detach().cpu(), "necessity": necessity.detach().cpu()},
        path,
    )
