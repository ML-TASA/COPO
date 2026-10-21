#!/usr/bin/env bash
# COPO release self-check: verify the vendored verl package and the COPO
# causal-reward / backend modules import and compile.
#
# This checks environment and code health only — it does NOT run any training or
# benchmark. Usage:
#
#     bash scripts/check.sh
#
# Optional: point PYTHON at a specific interpreter (default: python).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# Keep the repo free of .pyc artifacts: disable bytecode writing and redirect any
# that py_compile still emits to a throwaway cache removed on exit.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/copo_check_cache_$$"
trap 'rm -rf "$PYTHONPYCACHEPREFIX"' EXIT

status=0

echo "==> python: $("$PYTHON" --version 2>&1)"

echo "==> import checks (mirrors the installation sanity checks)"
if ! "$PYTHON" - "$ROOT_DIR" <<'PY'
import sys
import traceback

root = sys.argv[1]
sys.path.insert(0, root + "/verl")

checks = [
    ("verl + DataProto", "import verl; assert hasattr(verl, 'DataProto')"),
    ("causal_reward namespace",
     "from verl.utils.causal_reward import COPOConfig, apply_copo_advantage, build_causal_reward_tensor; "
     "from verl.utils.causal_reward import generator, generation, integration, precomputed, reward_manager; "
     "from verl.utils.causal_reward.collector import FixedTrajectoryCollector; "
     "from verl.utils.causal_reward.checkpoint import save_trainable_checkpoint; "
     "from verl.utils.causal_reward.backends import build_backend; "
     "assert COPOConfig is not None and callable(apply_copo_advantage) and callable(build_causal_reward_tensor)"),
    ("outcome scorer",
     "from verl.utils.reward_score.outcome import compute_score; assert callable(compute_score)"),
    ("backend implementations",
     "import verl.implementations.instructblip, verl.implementations.minigpt4, verl.implementations.qwen_vl"),
    ("legacy Qwen-VL runtime dependency",
     "from verl.implementations.qwen_vl import install_qwen_vl_compat; "
     "install_qwen_vl_compat(); import transformers_stream_generator"),
    ("training entries",
     "from verl.implementations.instructblip.train import main as a; "
     "from verl.implementations.minigpt4.train import main as b; assert callable(a) and callable(b)"),
    ("evaluation module",
     "from verl.evaluation.eval_pope import VALID_SPLITS, compute_metrics, normalize_answer; "
     "assert set(VALID_SPLITS) == {'default', 'random', 'popular', 'adversarial'}; "
     "assert callable(compute_metrics) and callable(normalize_answer)"),
]

failed = []
for name, stmt in checks:
    try:
        exec(stmt, {})
    except Exception:
        failed.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"  ok    {name}")

if failed:
    print(f"import checks FAILED: {len(failed)}/{len(checks)}")
    sys.exit(1)
print(f"import checks passed ({len(checks)}/{len(checks)})")
PY
then
    status=1
fi

echo "==> compile checks (core COPO modules)"
if ! "$PYTHON" -m py_compile \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/integration.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/collector.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/checkpoint.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/optimization.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/trajectory.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/generator.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/precomputed.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/reward_manager.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/backends/__init__.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/backends/hf_decoder.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/backends/llava15.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/backends/qwen_vl.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/backends/instructblip.py \
  "$ROOT_DIR"/verl/verl/utils/causal_reward/backends/minigpt4.py \
  "$ROOT_DIR"/verl/verl/trainer/ppo/core_algos.py \
  "$ROOT_DIR"/verl/verl/trainer/ppo/ray_trainer.py \
  "$ROOT_DIR"/verl/verl/model_merger/base_model_merger.py \
  "$ROOT_DIR"/verl/verl/experimental/agent_loop/single_turn_agent_loop.py \
  "$ROOT_DIR"/verl/verl/implementations/instructblip/train.py \
  "$ROOT_DIR"/verl/verl/implementations/instructblip/causal/causal_reward.py \
  "$ROOT_DIR"/verl/verl/implementations/instructblip/causal/causal_advantage.py \
  "$ROOT_DIR"/verl/verl/implementations/minigpt4/train.py \
  "$ROOT_DIR"/verl/verl/implementations/minigpt4/causal_reward.py \
  "$ROOT_DIR"/verl/verl/implementations/minigpt4/advantage.py \
  "$ROOT_DIR"/verl/verl/implementations/minigpt4/trainer.py \
  "$ROOT_DIR"/scripts/prepare_data/build_training_data.py \
  "$ROOT_DIR"/scripts/prepare_reward/generate_rewards.py
then
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "==> all checks passed"
else
    echo "==> checks FAILED (see above)"
fi
exit $status
