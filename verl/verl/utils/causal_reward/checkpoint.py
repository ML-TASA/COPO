"""Checkpoint helpers for trainable COPO adapters and optimizer state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_FORMAT = "copo.trainable_state.v1"


def _trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_trainable_checkpoint(
    path: str | Path,
    model,
    optimizer,
    *,
    step: int,
    model_family: str,
    config: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _trainable_state(model)
    if not state:
        raise ValueError("cannot checkpoint a model with no trainable parameters")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model_family": str(model_family),
            "step": int(step),
            "config": dict(config),
            "trainable_parameters": state,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_trainable_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    expected_model_family: str,
) -> int:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping) or value.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a supported COPO trainable checkpoint")
    if value.get("model_family") != expected_model_family:
        raise ValueError(
            f"checkpoint model family mismatch: {value.get('model_family')!r} != {expected_model_family!r}"
        )
    saved = value.get("trainable_parameters")
    if not isinstance(saved, Mapping) or not saved:
        raise ValueError("COPO checkpoint contains no trainable parameters")
    current = dict(model.named_parameters())
    expected = {name for name, parameter in current.items() if parameter.requires_grad}
    if set(saved) != expected:
        raise ValueError(
            f"checkpoint trainable parameter names mismatch: saved={sorted(saved)}, expected={sorted(expected)}"
        )
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = current[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(f"checkpoint shape mismatch for {name}")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    if optimizer is not None:
        optimizer_state = value.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)
    step = int(value.get("step", -1))
    if step < 0:
        raise ValueError("checkpoint training step must be non-negative")
    return step
