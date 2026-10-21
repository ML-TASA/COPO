#!/usr/bin/env bash
# COPO training — single entry point.
#
#     bash scripts/train.sh --model qwenvl \
#         --model-path /path/to/checkpoint \
#         --train-data /path/to/train.json \
#         --precomputed-reward /path/to/causal_rewards.pt
#
# ``--model`` selects a preset (llava15 / qwenvl / instructblip / minigpt4) that
# fills in the backend, GRPO group size, response length, learning rate and
# batch sizes. The Eq. 8 weights (lambda_s / lambda_n) come from the algorithm
# config. The user only supplies model / data / reward paths.
# Run with ``--help`` for all options.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m verl.implementations.launch "$@"
