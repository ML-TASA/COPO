"""Strict bridge from fixed v3 trajectories to model-specific adapter actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .trajectory import FixedTrajectoryBatch


@dataclass(frozen=True)
class FixedAdapterBatch:
    response_token_ids: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    outcome_rewards: torch.Tensor
    causal_rewards: torch.Tensor
    reasoning_mask: torch.Tensor
    answer_mask: torch.Tensor
    special_token_mask: torch.Tensor
    group_ids: tuple[tuple[str, str], ...]


def prepare_fixed_adapter_batch(
    selected: FixedTrajectoryBatch,
    *,
    prompt_token_ids: Sequence[int] | torch.Tensor,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    device: torch.device | str,
) -> FixedAdapterBatch:
    """Validate exact prompt/action identity and expose detached optimization data."""
    actual_ids = torch.as_tensor(response_token_ids).detach().cpu().long()
    actual_mask = torch.as_tensor(response_mask).detach().cpu().bool()
    stored_ids = selected._tensor("response_token_ids").cpu().long()
    stored_mask = selected._tensor("response_mask").cpu().bool()
    if actual_ids.shape != stored_ids.shape or not torch.equal(actual_ids, stored_ids):
        raise ValueError("fixed adapter response token IDs changed before policy evaluation")
    if actual_mask.shape != stored_mask.shape or not torch.equal(actual_mask, stored_mask):
        raise ValueError("fixed adapter response mask changed before policy evaluation")

    prompt = torch.as_tensor(prompt_token_ids).detach().cpu().long().reshape(-1)
    stored_prompts = selected._tensor("prompt_token_ids").cpu().long()
    prompt_lengths = selected._tensor("prompt_lengths").cpu().long()
    for row, length in enumerate(prompt_lengths.tolist()):
        if prompt.tolist() != stored_prompts[row, : int(length)].tolist():
            raise ValueError(f"fixed adapter prompt token mismatch at trajectory row {row}")

    target = torch.device(device)
    return FixedAdapterBatch(
        response_token_ids=stored_ids.to(target),
        response_mask=stored_mask.to(target),
        old_log_probs=selected._tensor("old_policy_log_probs").detach().to(target).float(),
        reference_log_probs=selected._tensor("reference_log_probs").detach().to(target).float(),
        outcome_rewards=selected._tensor("outcome_rewards").detach().to(target).float(),
        causal_rewards=selected._tensor("causal_reward").detach().to(target).float(),
        reasoning_mask=selected._tensor("reasoning_mask").detach().to(target).bool(),
        answer_mask=selected._tensor("answer_mask").detach().to(target).bool(),
        special_token_mask=selected._tensor("special_token_mask").detach().to(target).bool(),
        group_ids=tuple(
            (str(sample_id), str(group_id))
            for sample_id, group_id in zip(
                selected._list("sample_ids"), selected._list("group_ids"), strict=True
            )
        ),
    )


def compute_fixed_copo_advantages(
    batch: FixedAdapterBatch,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native-GRPO outcome advantage plus reasoning-only COPO Eq. 9 reward."""
    outcomes = batch.outcome_rewards
    scalar = torch.empty_like(outcomes)
    groups: dict[tuple[str, str], list[int]] = {}
    for row, group_id in enumerate(batch.group_ids):
        groups.setdefault(group_id, []).append(row)
    with torch.no_grad():
        for group_id, rows in groups.items():
            if len(rows) < 2:
                raise ValueError(f"fixed GRPO group {group_id!r} has fewer than two trajectories")
            values = outcomes[rows]
            std = values.std(unbiased=True)
            scalar[rows] = (values - values.mean()) / (std + epsilon)
        original = scalar.unsqueeze(1) * batch.response_mask
        causal = batch.causal_rewards * batch.reasoning_mask * batch.response_mask
        advantages = (original + causal) * batch.response_mask
    return advantages, original
