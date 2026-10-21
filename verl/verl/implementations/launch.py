"""Unified COPO training launcher.

Select a model and supply paths; the preset fills in every algorithm and
engineering parameter::

    bash scripts/train.sh --model qwenvl \\
        --model-path /path/to/checkpoint \\
        --train-data /path/to/train.json \\
        --precomputed-reward /path/to/causal_rewards.pt

The ``--model`` preset (see :mod:`verl.implementations.presets`) carries the
backend, GRPO group size, response length, learning rate and batch sizes. The
Eq. 8 weights (``lambda_s`` / ``lambda_n``) live in the algorithm config, not
the preset. Users only provide model / data / reward paths, which are
inherently per-user.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from verl.implementations.presets import LLAVA15, MODEL_PRESETS, QWENVL, get_preset


def _require_path(value: str, flag: str, env: str) -> str:
    if not value:
        raise SystemExit(f"--{flag} (or {env}) is required")
    if not os.path.exists(value):
        raise SystemExit(f"--{flag} path does not exist: {value}")
    return value


def _checkpoint_context_limit(model_path: str) -> int | None:
    """Read a local HF config without importing checkpoint remote code."""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read model context configuration {config_path}: {exc}") from exc

    def collect(mapping, names):
        values = []
        if not isinstance(mapping, dict):
            return values
        for key, value in mapping.items():
            if key in names and isinstance(value, int) and value > 0:
                values.append(value)
            elif isinstance(value, dict):
                values.extend(collect(value, names))
        return values

    positional = collect(config, {"max_position_embeddings", "n_positions"})
    if positional:
        return min(positional)
    fallbacks = collect(config, {"max_sequence_length", "seq_length", "model_max_length"})
    return min(fallbacks) if fallbacks else None


def _validate_context_budget(model_path: str, prompt_length: int, response_length: int) -> None:
    limit = _checkpoint_context_limit(model_path)
    if limit is not None and prompt_length + response_length > limit:
        raise SystemExit(
            f"configured prompt ({prompt_length}) + response ({response_length}) budget exceeds "
            f"checkpoint context limit ({limit}); choose smaller --max-prompt-length/"
            "--max-response-length values or a verified long-context checkpoint"
        )


def _agent_loop_worker_count(group_size: int, maximum: int = 8) -> int:
    """Choose a bounded worker count that divides every fixed trajectory group."""
    return max(value for value in range(1, min(group_size, maximum) + 1) if group_size % value == 0)


def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train COPO with a model preset.")
    p.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS),
                   help="Model preset that fills in all algorithm/engineering parameters.")
    p.add_argument("--model-path", default=None, help="Model/checkpoint path.")
    p.add_argument(
        "--train-data", default=None,
        help="Training dataset (.json/.parquet for native verl; JSONL for InstructBLIP/MiniGPT-4).",
    )
    p.add_argument("--val-data", default=None, help="Validation dataset (verl backends only).")
    p.add_argument("--precomputed-reward", default=None, help="Offline causal-reward .pt store.")
    p.add_argument("--output-dir", default=None, help="Output directory (defaults to outputs/<model>).")
    p.add_argument("--resume-from", default=None, help="InstructBLIP/MiniGPT-4 trainable-state checkpoint.")
    p.add_argument(
        "--training-steps", type=int, default=None,
        help="Required for InstructBLIP/MiniGPT-4; the paper does not define a 2/3-step training run.",
    )
    p.add_argument(
        "--training-epochs", type=int, default=None,
        help="Optional native-verl total epoch target; increase it when resuming with refreshed trajectories.",
    )
    p.add_argument(
        "--group-size", type=int, default=None,
        help="Override the model preset rollout group size (the v3 store must use the same value).",
    )
    p.add_argument(
        "--max-response-length", type=int, default=None,
        help="Override the preset response cap without changing the published preset.",
    )
    p.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override the released model preset learning rate (for example, 1e-6 from Appendix G).",
    )
    p.add_argument("--max-prompt-length", type=int, default=None,
                   help="Override the native-verl prompt cap without changing the preset.")
    p.add_argument("--train-batch-size", type=int, default=None,
                   help="Override the native-verl prompt batch size without changing the preset.")
    p.add_argument("--mini-batch-size", type=int, default=None,
                   help="Override the native-verl PPO mini-batch size without changing the preset.")
    p.add_argument(
        "--dtype", default=None, choices=["float16", "bfloat16", "float32"],
        help="InstructBLIP/MiniGPT-4 model dtype; defaults to the backend's established setting.",
    )
    p.add_argument(
        "--instructblip-protocol",
        default="original",
        choices=["original", "split_context_two_shot"],
        help="Optional InstructBLIP generation/scoring context; default: original.",
    )
    p.add_argument(
        "--micro-batch-size-per-gpu", type=int, default=1,
        help="Native verl actor/log-prob micro-batch size per GPU (default: 1).",
    )
    p.add_argument(
        "--rollout-gpu-memory-utilization", type=float, default=0.6,
        help="Native vLLM memory utilization fraction (default: 0.6).",
    )
    p.add_argument("--n-gpus-per-node", type=int,
                   default=int(os.environ.get("N_GPUS_PER_NODE", "8")), help="GPUs per node.")
    p.add_argument("--save-freq", type=int, default=20,
                   help="Native-verl checkpoint frequency in optimizer steps (default: 20).")
    # MiniGPT-4 additional weight sources (ignored by the other backends).
    p.add_argument("--llama-model", default=None, help="MiniGPT-4: Vicuna/LLaMA checkpoint (alias for --model-path).")
    p.add_argument("--vit-pretrained-path", default="", help="MiniGPT-4: BLIP-2 eva_vit_g.pth.")
    p.add_argument("--qformer-model", default="", help="MiniGPT-4: BLIP-2 Q-Former checkpoint.")
    p.add_argument(
        "--ckpt", default="",
        help="MiniGPT-4: stage-2 checkpoint matching the selected 7B/13B language-model base.",
    )
    p.add_argument("--bert-model-path", default="", help="MiniGPT-4: bert-base-uncased snapshot.")
    return p


def _verl_command(args: argparse.Namespace, preset: dict, root: str) -> list[str]:
    model = _require_path(args.model_path or os.environ.get("MODEL_PATH", ""),
                          "model-path", "MODEL_PATH")
    train = _require_path(args.train_data or os.environ.get("TRAIN_DATA", ""),
                          "train-data", "TRAIN_DATA")
    val = args.val_data or os.environ.get("VAL_DATA") or train
    reward = _require_path(args.precomputed_reward or os.environ.get("PRECOMPUTED_REWARD", ""),
                           "precomputed-reward", "PRECOMPUTED_REWARD")
    out = args.output_dir or os.path.join(root, "outputs", f"{args.model}_train")
    total_epochs = args.training_epochs if args.training_epochs is not None else preset["total_epochs"]
    group_size = args.group_size if args.group_size is not None else preset["group_size"]
    max_response_length = (
        args.max_response_length
        if args.max_response_length is not None
        else preset["max_response_length"]
    )
    max_prompt_length = (
        args.max_prompt_length if args.max_prompt_length is not None else preset["max_prompt_length"]
    )
    train_batch_size = (
        args.train_batch_size if args.train_batch_size is not None else preset["train_batch_size"]
    )
    mini_batch_size = (
        args.mini_batch_size if args.mini_batch_size is not None else preset["mini_batch_size"]
    )
    learning_rate = args.learning_rate if args.learning_rate is not None else preset["lr"]
    if total_epochs < 1:
        raise SystemExit("--training-epochs must be positive")
    if group_size < 2:
        raise SystemExit("--group-size must be at least 2 for GRPO")
    if max_response_length < 1:
        raise SystemExit("--max-response-length must be positive")
    if max_prompt_length < 1 or train_batch_size < 1 or mini_batch_size < 1:
        raise SystemExit("--max-prompt-length, --train-batch-size, and --mini-batch-size must be positive")
    if learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.micro_batch_size_per_gpu < 1:
        raise SystemExit("--micro-batch-size-per-gpu must be positive")
    if not 0 < args.rollout_gpu_memory_utilization < 1:
        raise SystemExit("--rollout-gpu-memory-utilization must be between 0 and 1")
    if args.save_freq < 1:
        raise SystemExit("--save-freq must be positive")
    _validate_context_budget(model, max_prompt_length, max_response_length)

    command = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.name=copo",
        "algorithm.adv_estimator=grpo",
        "algorithm.copo.reward_source=precomputed",
        f"algorithm.copo.precomputed_reward_path={reward}",
        f"data.train_files={train}",
        f"data.val_files={val}",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation='error'",
        "data.image_key=images",
        f"actor_rollout_ref.model.path={model}",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        # Both paper VLMs use model-specific forward paths which are not
        # compatible with verl's text-only padding removal / fused kernels.
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.use_fused_kernels=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_batch_size}",
        # The worker normalizes ppo_mini_batch_size by rollout.n / world_size.
        # One is valid for every supported GPU count and is the only setting
        # historically exercised with the 8k multimodal response budget.
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={preset['kl_loss_coef']}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.load_format=hf",
        "actor_rollout_ref.rollout.layered_summon=False",
        f"actor_rollout_ref.rollout.n={group_size}",
        f"actor_rollout_ref.rollout.agent.num_workers={_agent_loop_worker_count(group_size)}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        'trainer.logger=["console","wandb"]',
        "trainer.project_name='copo'",
        f"trainer.experiment_name={args.model}",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        "trainer.nnodes=1",
        f"trainer.save_freq={args.save_freq}",
        "trainer.test_freq=-1",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.default_local_dir={out}",
    ]

    if args.model == QWENVL:
        # Qwen-VL-Chat (2023) is a legacy remote-code checkpoint. These
        # settings are required by its tokenizer/forward wrapper and by vLLM's
        # native Qwen-VL multimodal implementation.
        command.extend(
            [
                "data.trust_remote_code=True",
                "actor_rollout_ref.model.trust_remote_code=True",
                "actor_rollout_ref.model.lora_rank=8",
                "actor_rollout_ref.model.lora_alpha=16",
                r"actor_rollout_ref.model.target_modules='.*transformer\.h\..*\.(c_attn|c_proj|w1|w2)'",
                "+actor_rollout_ref.model.override_config.attn_implementation=eager",
                "+actor_rollout_ref.model.override_config.num_key_value_heads=32",
                "actor_rollout_ref.rollout.free_cache_engine=False",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.architectures='[\"QwenVLForConditionalGeneration\"]'",
                "algorithm.copo.fixed_trajectory=True",
                "algorithm.copo.fixed_model_family=qwenvl",
                "actor_rollout_ref.actor.policy_loss.loss_mode=copo_eq11",
                "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
                "+actor_rollout_ref.actor.use_rollout_log_probs=True",
                "trainer.val_before_train=False",
            ]
        )
    elif args.model == LLAVA15:
        command.extend(
            [
                "algorithm.copo.fixed_trajectory=True",
                "algorithm.copo.fixed_model_family=llava15",
                "actor_rollout_ref.actor.policy_loss.loss_mode=copo_eq11",
                "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
                "+actor_rollout_ref.actor.use_rollout_log_probs=True",
                "trainer.val_before_train=False",
                "actor_rollout_ref.model.lora_rank=8",
                "actor_rollout_ref.model.lora_alpha=16",
                r"actor_rollout_ref.model.target_modules='.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'",
                "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
                "actor_rollout_ref.rollout.free_cache_engine=True",
            ]
        )
    else:  # defensive: _verl_command is only valid for the two verl presets
        raise ValueError(f"unsupported verl model preset: {args.model}")

    return command


def _adapter_command(args: argparse.Namespace, preset: dict, root: str) -> list[str]:
    backend = preset["backend"]
    out = args.output_dir or os.path.join(root, "outputs", f"{args.model}_train")
    group_size = args.group_size if args.group_size is not None else preset["group_size"]
    max_response_length = (
        args.max_response_length
        if args.max_response_length is not None
        else preset["max_response_length"]
    )
    learning_rate = args.learning_rate if args.learning_rate is not None else preset["lr"]
    if group_size < 2:
        raise SystemExit("--group-size must be at least 2 for GRPO")
    if max_response_length < 1:
        raise SystemExit("--max-response-length must be positive")
    if learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.training_steps is None or args.training_steps < 1:
        raise SystemExit(
            "--training-steps must be a positive explicit value for InstructBLIP/MiniGPT-4; "
            "the former 3/2-step values were interface checks, not paper training schedules"
        )
    resume_from = None
    if args.resume_from:
        resume_from = _require_path(args.resume_from, "resume-from", "RESUME_FROM")

    if backend == "instructblip":
        model = _require_path(args.model_path or os.environ.get("MODEL_PATH", ""),
                              "model-path", "MODEL_PATH")
        dataset = _require_path(args.train_data or os.environ.get("DATASET", ""),
                                "train-data", "DATASET")
        reward = _require_path(args.precomputed_reward or os.environ.get("PRECOMPUTED_REWARD", ""),
                               "precomputed-reward", "PRECOMPUTED_REWARD")
        return [
            sys.executable, "-m", "verl.implementations.instructblip.train",
            "--model", model,
            "--dataset", dataset,
            "--causal-reward", reward,
            "--fixed-trajectory",
            "--group-size", str(group_size),
            "--max-new-tokens", str(max_response_length),
            "--lr", str(learning_rate),
            "--clip-ratio", str(preset["clip_ratio"]),
            "--kl-coef", str(preset["kl_loss_coef"]),
            "--steps", str(args.training_steps),
            "--output-dir", out,
            "--instructblip-protocol", getattr(args, "instructblip_protocol", "original"),
        ] + (["--dtype", args.dtype] if args.dtype else []) \
          + (["--resume-from", resume_from] if resume_from else [])

    # minigpt4
    llama = _require_path(args.llama_model or args.model_path or os.environ.get("LLAMA_MODEL", ""),
                          "model-path", "LLAMA_MODEL")
    vit = _require_path(args.vit_pretrained_path or os.environ.get("VIT_PRETRAINED_PATH", ""),
                        "vit-pretrained-path", "VIT_PRETRAINED_PATH")
    qformer = _require_path(args.qformer_model or os.environ.get("QFORMER_MODEL", ""),
                            "qformer-model", "QFORMER_MODEL")
    stage2 = _require_path(args.ckpt or os.environ.get("MINIGPT4_CKPT", ""),
                           "ckpt", "MINIGPT4_CKPT")
    bert = _require_path(args.bert_model_path or os.environ.get("BERT_MODEL_PATH", ""),
                         "bert-model-path", "BERT_MODEL_PATH")
    dataset = _require_path(args.train_data or os.environ.get("DATASET", ""),
                            "train-data", "DATASET")
    reward = _require_path(args.precomputed_reward or os.environ.get("PRECOMPUTED_REWARD", ""),
                           "precomputed-reward", "PRECOMPUTED_REWARD")
    return [
        sys.executable, "-m", "verl.implementations.minigpt4.train",
        "--llama-model", llama,
        "--vit-pretrained-path", vit,
        "--qformer-model", qformer,
        "--ckpt", stage2,
        "--bert-model-path", bert,
        "--dataset", dataset,
        "--causal-reward", reward,
        "--fixed-trajectory",
        "--group-size", str(group_size),
        "--grpo-max-new-tokens", str(max_response_length),
        "--lr", str(learning_rate),
        "--clip-ratio", str(preset["clip_ratio"]),
        "--kl-coef", str(preset["kl_loss_coef"]),
        "--grpo-steps", str(args.training_steps),
        "--output-dir", out,
    ] + (["--dtype", args.dtype] if args.dtype else []) \
      + (["--resume-from", resume_from] if resume_from else [])


def main(argv: Sequence[str] | None = None) -> int:
    args = _base_parser().parse_args(argv)
    preset = get_preset(args.model)
    root = str(Path(__file__).resolve().parents[3])

    if preset["backend"] == "verl":
        command = _verl_command(args, preset, root)
    else:
        command = _adapter_command(args, preset, root)

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("HF_HUB_OFFLINE", "1")

    print("COPO train:", " ".join(command))
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
