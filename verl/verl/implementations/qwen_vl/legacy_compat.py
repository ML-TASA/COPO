"""Import compatibility for Qwen-VL-Chat's legacy transformer dependencies."""

from __future__ import annotations

import importlib
from typing import Iterable

import transformers

_BEAM_SYMBOLS = ("BeamSearchScorer", "BeamHypotheses", "BeamScorer")
_BEAM_SOURCE = "transformers.generation.beam_search"


def _missing_symbols() -> Iterable[str]:
    for name in _BEAM_SYMBOLS:
        if not hasattr(transformers, name):
            yield name


def install_qwen_vl_compat() -> bool:
    """Ensure legacy beam-search symbols exist on top-level ``transformers``.

    Returns ``True`` if anything was installed, ``False`` if everything was
    already present.  Idempotent; safe to call more than once.
    """
    missing = list(_missing_symbols())
    if not missing:
        return False

    beam_search = importlib.import_module(_BEAM_SOURCE)
    for name in missing:
        if not hasattr(beam_search, name):
            raise ImportError(
                f"Qwen-VL legacy compat: expected '{name}' in {_BEAM_SOURCE} "
                f"(transformers {transformers.__version__}) but it is absent."
            )
        setattr(transformers, name, getattr(beam_search, name))
    return True
