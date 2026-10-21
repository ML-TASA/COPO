"""Minimal re-implementations of the ``timm.models.layers`` helpers used by the vendored ``eva_vit.py``."""

from __future__ import annotations

import torch


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Stochastic depth (DropPath) — matches timm's ``drop_path`` semantics."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


trunc_normal_ = torch.nn.init.trunc_normal_
