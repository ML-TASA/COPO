"""Fixed-trajectory COPO training entry point for MiniGPT-4.

Usage::

    python -m verl.implementations.minigpt4.train \
        --llama-model /path/to/vicuna-7b \
        --vit-pretrained-path /path/to/eva_vit_g.pth \
        --qformer-model /path/to/blip2_pretrained_flant5xxl.pth \
        --ckpt /path/to/pretrained_minigpt4.pth \
        --bert-model-path /path/to/bert-base-uncased \
        --dataset /path/to/samples.jsonl \
        --causal-reward /path/to/causal_rewards.pt \
        --fixed-trajectory \
        --output-dir outputs/minigpt4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import torch
from verl.utils.causal_reward.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint
from verl.utils.causal_reward.collector import model_state_fingerprint

from verl.utils.causal_reward.trajectory import (
    FixedTrajectoryStore,
    fingerprint_mapping,
    fingerprint_model_config,
    fingerprint_tokenizer,
)

from .config import MiniGPT4Config
from .trainer import COPOTrainer
from .data import MiniGPT4Sample, load_image
from .model import apply_freeze, load_model
from .processor import MiniGPT4Processor
from .reward import OutcomeReward

logger = logging.getLogger(__name__)


def load_samples(path: str, *, fixed_trajectory: bool = False) -> list[MiniGPT4Sample]:
    """Load ``(image, instruction, answer)`` samples from a JSONL file.

    Each line is ``{"sample_id": "<stable-id>", "image": "<path>",
    "instruction": "<question>", "answer": "<ground-truth>", "task": "vqa"}``.
    ``answer`` is the ground-truth scored by the outcome reward. The exact
    response actions are loaded from the required v3 trajectory store; an
    optional dataset ``response`` field is ignored.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")
    samples: list[MiniGPT4Sample] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_no} of {path}: {exc}") from exc
        for key in ("sample_id", "image", "instruction"):
            if key not in item:
                raise ValueError(f"line {line_no} of {path} is missing required field {key!r}")
        if fixed_trajectory:
            for key in ("group_id", "image_id"):
                if not str(item.get(key, "")).strip():
                    raise ValueError(
                        f"line {line_no} of {path} is missing fixed-trajectory identity field {key!r}"
                    )
        samples.append(MiniGPT4Sample(
            sample_id=str(item["sample_id"]),
            image_path=item["image"],
            instruction=item["instruction"],
            answer=item.get("answer", ""),
            task=item.get("task", "vqa"),
            group_id=str(item.get("group_id", "")),
            image_id=str(item.get("image_id", "")),
        ))
    return samples


def build_config(args: argparse.Namespace) -> MiniGPT4Config:
    return MiniGPT4Config(
        vit_model="eva_clip_g",
        vit_pretrained_path=args.vit_pretrained_path,
        q_former_model=args.qformer_model,
        llama_model=args.llama_model,
        ckpt=args.ckpt,
        bert_model_path=args.bert_model_path,
        dtype=args.dtype,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        group_size=args.group_size,
        grpo_steps=args.grpo_steps,
        grpo_max_new_tokens=args.grpo_max_new_tokens,
        clip_ratio=args.clip_ratio,
        kl_loss_coef=args.kl_coef,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train COPO with the MiniGPT-4 backend.")
    parser.add_argument("--llama-model", required=True, help="Vicuna/LLaMA HF checkpoint directory")
    parser.add_argument("--vit-pretrained-path", default="", help="BLIP-2 eva_vit_g.pth")
    parser.add_argument("--qformer-model", default="", help="BLIP-2 blip2_pretrained_flant5xxl.pth")
    parser.add_argument(
        "--ckpt", default="",
        help="MiniGPT-4 stage-2 checkpoint matching the selected 7B/13B language-model base",
    )
    parser.add_argument("--bert-model-path", default="", help="bert-base-uncased snapshot directory")
    parser.add_argument(
        "--dataset", required=True,
        help="JSONL of {sample_id, image, instruction, answer, task?} samples",
    )
    parser.add_argument("--causal-reward", required=True, help="copo.fixed_trajectory.v3 .pt store")
    parser.add_argument(
        "--fixed-trajectory",
        action="store_true",
        required=True,
        help="required: consume a copo.fixed_trajectory.v3 store",
    )
    parser.add_argument("--lambda-s", type=float, default=0.35, help="sufficiency weight (COPO Eq. 8)")
    parser.add_argument("--lambda-n", type=float, default=0.35, help="necessity weight (COPO Eq. 8)")
    parser.add_argument("--group-size", type=int, default=8, help="responses sampled per prompt")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--grpo-steps", type=int, required=True, help="fixed-trajectory update iterations")
    parser.add_argument("--grpo-max-new-tokens", type=int, default=2048, help="stored response length limit")
    parser.add_argument(
        "--outcome-reward",
        default="answer",
        choices=["answer"],
        help="ground-truth task outcome reward required by fixed COPO training",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto", help="auto -> cuda if available else cpu")
    parser.add_argument("--output-dir", default="outputs/minigpt4", help="where to save trainable weights")
    parser.add_argument("--resume-from", default=None, help="resume trainable parameters, optimizer, and step")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.lambda_s != 0.35 or args.lambda_n != 0.35:
        raise ValueError("fixed COPO training requires lambda_s=lambda_n=0.35")

    config = build_config(args)
    config.validate()

    model = load_model(config)
    model = model.to(config.resolve_device())
    apply_freeze(model, config)

    processor = MiniGPT4Processor(config)
    samples = load_samples(args.dataset, fixed_trajectory=args.fixed_trajectory)
    if not samples:
        raise ValueError(f"no samples loaded from {args.dataset}")

    answers = [s.answer for s in samples]
    if any(not str(answer).strip() for answer in answers):
        raise ValueError(
            "--outcome-reward answer requires a nonempty 'answer' field in every "
            "dataset line; add the task's real ground-truth answers"
        )
    outcome_backend = OutcomeReward()
    tasks = [s.task for s in samples]

    tokenizer = model.llama_tokenizer
    eos_ids = tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    if not eos_ids or tokenizer.pad_token_id is None:
        raise ValueError("fixed MiniGPT-4 training requires explicit EOS and padding token IDs")
    processor_fingerprint = fingerprint_mapping(
        {
            "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
            "prompt_template": config.prompt_template,
            "image_size": config.image_size,
        }
    )
    model_fingerprint = fingerprint_mapping(
        {
            "llama_config": fingerprint_model_config(config.llama_model),
            "vit_model": config.vit_model,
            "image_size": config.image_size,
            "num_query_token": config.num_query_token,
            "has_qformer": config.has_qformer,
            "prompt_template": config.prompt_template,
            "end_sym": config.end_sym,
        }
    )
    expected_metadata = {
            "model_family": "minigpt4",
            "tokenizer_fingerprint": fingerprint_tokenizer(tokenizer),
            "processor_template_fingerprint": processor_fingerprint,
            "model_config_fingerprint": model_fingerprint,
            "group_size": config.group_size,
            "max_response_length": config.grpo_max_new_tokens,
            "lambda_s": 0.35,
            "lambda_n": 0.35,
            "eos_token_ids": [int(token) for token in eos_ids],
            "pad_token_id": int(tokenizer.pad_token_id),
    }
    if not args.resume_from:
        expected_metadata["collector_policy_id"] = model_state_fingerprint(model)
    fixed_store = FixedTrajectoryStore.load(
        args.causal_reward,
        expected_metadata=expected_metadata,
    )
    trainer = COPOTrainer(
        model,
        config,
        processor,
        outcome_backend,
        None,
        answers=answers,
        tasks=tasks,
        sample_ids=[s.sample_id for s in samples],
        fixed_store=fixed_store,
        group_ids=[s.group_id for s in samples],
        image_ids=[s.image_id for s in samples],
    )

    images = [load_image(s.image_path) for s in samples]
    instructions = [s.instruction for s in samples]

    start_step = 0
    if args.resume_from:
        start_step = load_trainable_checkpoint(
            args.resume_from,
            model,
            trainer.optimizer,
            expected_model_family="minigpt4",
        )
        if start_step >= config.grpo_steps:
            raise ValueError(
                f"resume step {start_step} is not smaller than configured steps {config.grpo_steps}"
            )

    logger.info("starting COPO training: %d samples, group_size=%d, steps=%d",
                len(samples), config.group_size, config.grpo_steps)
    for step in range(start_step, config.grpo_steps):
        stats = trainer.step(images, instructions)
        logger.info("step %d/%d  loss=%.4f  grad_norm=%.4f",
                    step + 1, config.grpo_steps, stats["loss"], stats["grad_norm"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    torch.save(trainable, output_dir / "minigpt4_llama_proj.pt")
    save_trainable_checkpoint(
        output_dir / "minigpt4_training_state.pt",
        model,
        trainer.optimizer,
        step=config.grpo_steps,
        model_family="minigpt4",
        config={
            "group_size": config.group_size,
            "max_response_length": config.grpo_max_new_tokens,
            "lr": config.lr,
            "clip_ratio": config.clip_ratio,
            "kl_loss_coef": config.kl_loss_coef,
        },
    )
    logger.info("saved trainable projection weights to %s", output_dir / "minigpt4_llama_proj.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
