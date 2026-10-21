"""Pytest configuration for the vendored ``verl`` package.

Kept in ``verl/`` rather than at the repository root so the top-level directory
stays clean. pytest's default "prepend" import mode puts this directory on
``sys.path``, which is all that is needed to ``import verl`` from the tests under
``verl/verl/.../tests``; the explicit insert below keeps that working under any
import mode.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
