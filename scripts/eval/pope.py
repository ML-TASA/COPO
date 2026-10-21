"""Run POPE hallucination evaluation on a COPO checkpoint.

Thin CLI over :mod:`verl.evaluation.eval_pope`. Run from anywhere — the vendored
``verl/`` package is added to ``sys.path`` automatically:

    python scripts/eval/pope.py --checkpoint /path/to/checkpoint --split adversarial
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "verl"))

from verl.evaluation.eval_pope import main  # noqa: E402

if __name__ == "__main__":
    main()
