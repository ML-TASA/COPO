"""Collect strict fixed trajectories consumed by COPO fixed training.

Thin CLI over :mod:`verl.utils.causal_reward.generator`. Run from anywhere — the
vendored ``verl/`` package is added to ``sys.path`` automatically:

    python scripts/prepare_reward/generate_rewards.py \\
        --model qwenvl \\
        --model-path /path/to/base_model \\
        --reference-model /path/to/reference_policy \\
        --dataset /path/to/samples.jsonl \\
        --output /path/to/causal_rewards.pt

``--model`` selects one of the four formal experiment backends. The collector
stores raw action IDs, old/reference log probabilities, outcome and causal
rewards under the strict ``copo.fixed_trajectory.v3`` contract. Checkpoints
must exist locally; the CLI never substitutes a text-only model.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "verl"))

from verl.utils.causal_reward.generator import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
