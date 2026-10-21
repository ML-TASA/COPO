"""Input preparation for Qwen-VL-Chat (2023)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union


def _qwen_vl_image_token_ids(tokenizer) -> Optional[tuple[int, int, int]]:
    """Return ``(img_start_id, img_pad_id, img_end_id)`` or ``None`` if not Qwen-VL."""
    img_start = getattr(tokenizer, "img_start_id", None)
    img_pad = getattr(tokenizer, "img_pad_id", None)
    img_end = getattr(tokenizer, "img_end_id", None)
    if img_start is None or img_pad is None or img_end is None:
        return None
    return int(img_start), int(img_pad), int(img_end)


def dedup_qwen_vl_image_tokens(
    prompt_ids: Sequence[int],
    tokenizer,
) -> list[int]:
    """Collapse padded ``[img_start, img_pad*..., img_end]`` to ``[img_start, img_end]``.

    Only operates on Qwen-VL's tokenizer; returns ``prompt_ids`` unchanged
    otherwise.
    """
    ids = _qwen_vl_image_token_ids(tokenizer)
    if ids is None:
        return list(prompt_ids)

    img_start, img_pad, img_end = ids
    tokens = list(prompt_ids)
    out: List[int] = []
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i] == img_start:
            j = i + 1
            while j < n and tokens[j] == img_pad:
                j += 1
            if j < n and tokens[j] == img_end:
                # Valid <img> span; emit the unpadded [img_start, img_end].
                out.append(img_start)
                out.append(img_end)
                i = j + 1
                continue
        out.append(tokens[i])
        i += 1
    return out


def resolve_image_path(image_ref: Union[str, os.PathLike], image_root: Optional[Union[str, os.PathLike]] = None) -> str:
    """Resolve a relative image reference against ``image_root``."""
    path = Path(image_ref)
    if path.is_absolute():
        return str(path)
    if image_root is None:
        raise ValueError(
            "Qwen-VL image reference is relative but no image_root was provided; "
            "pass --image-root or store absolute paths."
        )
    return str((Path(image_root) / path).resolve())
