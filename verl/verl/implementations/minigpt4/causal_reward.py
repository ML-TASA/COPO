"""Token-level offline causal reward for the MiniGPT-4 COPO stage."""

from __future__ import annotations

import torch

from verl.utils.causal_reward.precomputed import match_precomputed_trajectory_rows


def combine_sufficiency_necessity(sufficiency, necessity, lambda_s: float = 0.35, lambda_n: float = 0.35):
    """Combine token-level sufficiency and necessity into the COPO causal reward.

    Reproduces COPO Eq. (8): ``r_causal = lambda_s * S_suff + lambda_n * S_nec``
    (weighted sum, ``lambda_s = lambda_n = 0.35``).  Only used as a fallback when
    the store carries no precomputed ``causal_reward``.
    """
    return lambda_s * sufficiency + lambda_n * necessity


def load_causal_rewards(path: str) -> dict[str, torch.Tensor]:
    """Load a unified ``{"sufficiency", "necessity", "causal_reward"}`` store.

    The legacy ``suff`` / ``nec`` keys are accepted for backwards compatibility;
    newly generated stores use the unified key names.
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("offline causal reward store must be a dict")
    if "sufficiency" not in data and "suff" in data:
        data = dict(data)
        data["sufficiency"] = data.pop("suff")
        if "nec" in data and "necessity" not in data:
            data["necessity"] = data.pop("nec")
    return data


class OfflineCausalRewardProvider:
    """Provide token-level causal reward aligned to a response mask.

    The store is ``(num_samples * num_rollouts, response_length)``; the caller
    passes ``row_start`` to select the rows for the current sample's group and
    the tensors are truncated to the response-mask length.
    """

    def __init__(self, path: str, lambda_s: float = 0.35, lambda_n: float = 0.35,
                 num_samples: int | None = None, group_size: int | None = None):
        self.path = path
        self.lambda_s = lambda_s
        self.lambda_n = lambda_n
        self.num_samples = num_samples
        self.group_size = group_size
        self._store = None

    def _load(self) -> dict[str, torch.Tensor]:
        if self._store is None:
            self._store = load_causal_rewards(self.path)
            self._validate_row_count(self._store)
        return self._store

    def _validate_row_count(self, store: dict[str, torch.Tensor]) -> None:
        """Reject a store whose rollout count does not match ``num_samples * group_size``.

        A store generated with a different rollout count would otherwise be sliced
        by row index and silently misalign the sample -> rollout mapping.
        """
        if self.num_samples is None or self.group_size is None:
            return
        expected = self.num_samples * self.group_size
        for key in ("causal_reward", "sufficiency", "necessity"):
            tensor = store.get(key)
            if tensor is None:
                continue
            actual = torch.as_tensor(tensor).shape[0]
            if actual != expected:
                raise ValueError(
                    f"causal reward store has {actual} rows, expected {expected} "
                    f"({self.num_samples} samples x {self.group_size} rollouts); "
                    f"regenerate it with a matching rollout count"
                )
            return

    def get_causal_reward(self, response_mask: torch.Tensor, response_token_ids=None,
                          sample_ids=None) -> torch.Tensor:
        """Return a ``(B, L)`` token-level causal reward masked to ``response_mask``."""
        store = self._load()
        batch, length = response_mask.shape
        rows = match_precomputed_trajectory_rows(
            store, response_token_ids, response_mask, sample_ids
        )

        def take(key):
            tensor = store.get(key)
            if tensor is None:
                return None
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
            if tensor.dim() != 2:
                raise ValueError(f"{key} must be a 2D (samples, response_length) tensor")
            if rows and tensor.shape[0] <= max(rows):
                raise ValueError(f"offline causal reward store has too few {key} rows")
            if tensor.shape[1] < length:
                raise ValueError(
                    f"{key} response length {tensor.shape[1]} < response mask length {length}"
                )
            return tensor[rows, :length]

        causal = take("causal_reward")
        if causal is None:
            sufficiency = take("sufficiency")
            necessity = take("necessity")
            if sufficiency is None or necessity is None:
                raise ValueError(
                    "offline causal reward store must contain causal_reward or "
                    "sufficiency + necessity"
                )
            causal = combine_sufficiency_necessity(sufficiency, necessity, self.lambda_s, self.lambda_n)
        return causal.to(response_mask.device) * response_mask.float()
