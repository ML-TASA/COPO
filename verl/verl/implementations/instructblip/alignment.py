"""Response-only alignment helpers (pure tensor ops, no model dependency).

These functions map a full-sequence ``(image + instruction + response)``
log-prob / mask pair onto the response span and pad it to the offline
causal-reward store width.  Kept dependency-free (``torch`` only) so the
alignment logic can be unit-tested without loading a model.
"""

from __future__ import annotations

import torch


def append_response_tokens(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_length: int,
):
    """Append the exact sampled continuation and build response-only labels."""
    if prompt_input_ids.shape != prompt_attention_mask.shape or prompt_input_ids.shape[0] != 1:
        raise ValueError("prompt input_ids and attention_mask must have identical single-row shapes")
    response = torch.as_tensor(response_ids, dtype=prompt_input_ids.dtype, device=prompt_input_ids.device).reshape(1, -1)
    if response.shape[1] > int(response_length):
        raise ValueError(
            f"sampled response has {response.shape[1]} tokens, exceeding response_length={response_length}"
        )
    input_ids = torch.cat([prompt_input_ids, response], dim=1)
    response_attention = torch.ones_like(response)
    attention_mask = torch.cat([prompt_attention_mask, response_attention], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_input_ids.shape[1] :] = response
    return input_ids, attention_mask, labels


def extract_response_only(gathered: torch.Tensor, response_mask: torch.Tensor, response_length: int):
    """Slice the response span out of the full sequence, padded to ``response_length``.

    ``gathered`` and ``response_mask`` are the full-sequence ``(B, T-1)``
    log-probs / mask from the causal-shift step: index ``i`` describes full
    token ``i+1``, and ``response_mask`` is True only on response tokens.  The
    response tokens are collected in order and padded to a fixed
    ``response_length`` so the result aligns 1:1 with the offline causal-reward
    store (also ``(B, response_length)``).

    Returns ``(log_probs, mask)``, both ``(B, response_length)``.
    """
    length = int(response_length)
    actual_lengths = response_mask.long().sum(dim=1)
    if torch.any(actual_lengths > length):
        rows = torch.where(actual_lengths > length)[0].tolist()
        raise ValueError(
            f"response tokens exceed configured response_length={length} for rows {rows}; "
            "refusing to truncate token log-probs because that would break causal-reward alignment"
        )
    offset = (response_mask.long().cumsum(dim=1) - 1).clamp(min=0)  # (B, T-1): 0-based response index
    valid = response_mask
    out_logp = torch.zeros(response_mask.shape[0], length, dtype=gathered.dtype, device=gathered.device)
    out_mask = torch.zeros(response_mask.shape[0], length, dtype=torch.bool, device=gathered.device)
    row, col = valid.nonzero(as_tuple=True)
    out_logp[row, offset[row, col]] = gathered[row, col]
    out_mask[row, offset[row, col]] = True
    return out_logp, out_mask
