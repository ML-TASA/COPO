"""Fixed-trajectory COPO training entry point for InstructBLIP.

Usage::

    python -m verl.implementations.instructblip.train \
        --model /path/to/instructblip \
        --dataset /path/to/samples.jsonl \
        --causal-reward /path/to/causal_rewards.pt \
        --fixed-trajectory \
        --output-dir outputs/instructblip
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import torch

from verl.utils.causal_reward.adapter import compute_fixed_copo_advantages, prepare_fixed_adapter_batch
from verl.utils.causal_reward.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint
from verl.utils.causal_reward.collector import model_state_fingerprint
from verl.utils.causal_reward.backends.hf_decoder import generation_eos_token_ids
from verl.utils.causal_reward.optimization import compute_copo_eq11_total_loss
from verl.utils.causal_reward.trajectory import (
    FixedTrajectoryStore,
    fingerprint_model_config,
    fingerprint_tokenizer,
    multimodal_content_hash,
)

from .config import InstructBlipConfig
from .data import InstructBlipSample, load_image
from .model import apply_freeze, build_optimizer, load_model, response_log_probs
from .processor import (
    InstructBlipProcessor,
    ORIGINAL_PROTOCOL,
    SUPPORTED_PROTOCOLS,
    protocol_processor_fingerprint,
    render_prompt_context,
)
from .reward import OutcomeReward

logger = logging.getLogger(__name__)


def validate_store_protocol(fixed_store: FixedTrajectoryStore, requested_protocol: str) -> None:
    """Reject Original/Candidate store mixing before any optimizer update."""
    stored_protocol = fixed_store.metadata["sampling_parameters"].get(
        "protocol_id", ORIGINAL_PROTOCOL
    )
    if stored_protocol != requested_protocol:
        raise ValueError(
            "fixed InstructBLIP generation protocol mismatch: "
            f"stored={stored_protocol!r}, requested={requested_protocol!r}"
        )


def load_samples(path: str, *, fixed_trajectory: bool = False) -> list[InstructBlipSample]:
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
    samples: list[InstructBlipSample] = []
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
        samples.append(InstructBlipSample(
            sample_id=str(item["sample_id"]),
            image_path=item["image"],
            instruction=item["instruction"],
            answer=item.get("answer", ""),
            task=item.get("task", "vqa"),
            group_id=str(item.get("group_id", "")),
            image_id=str(item.get("image_id", "")),
        ))
    return samples


class InstructBlipCOPOTrainer:
    """Fixed-trajectory Eq. 11 update loop for the InstructBLIP backend."""

    def __init__(self, model, config: InstructBlipConfig, processor, outcome_backend, causal_provider=None,
                 answers=None, tasks=None, sample_ids=None, fixed_store=None,
                 group_ids=None, image_ids=None):
        self.model = model
        self.config = config
        self.processor = processor
        self.outcome_backend = outcome_backend
        self.causal_provider = causal_provider
        self.optimizer = build_optimizer(model, config)
        self.group_size = config.batch_size
        self.answers = list(answers) if answers is not None else None
        self.tasks = list(tasks) if tasks is not None else None
        self.sample_ids = list(sample_ids) if sample_ids is not None else None
        self.fixed_store = fixed_store
        self.group_ids = list(group_ids) if group_ids is not None else None
        self.image_ids = list(image_ids) if image_ids is not None else None

    def step(self, images, instructions):
        """Run one strict fixed-trajectory Eq. 11 update."""
        contexts = [
            render_prompt_context(instruction, self.config.generation_protocol)
            for instruction in instructions
        ]
        if self.fixed_store is not None:
            return self._fixed_step(images, contexts)
        raise RuntimeError(
            "formal InstructBLIP COPO training requires a validated "
            "copo.fixed_trajectory.v3 store; the legacy rollout policy-gradient "
            "path is not the Eq. 11 objective"
        )

    def _fixed_step(self, images, contexts):
        """One Eq. 11 update over exact actions from a validated v3 store."""
        if not all(values is not None for values in (self.sample_ids, self.group_ids, self.image_ids)):
            raise ValueError("fixed InstructBLIP training requires sample/group/image identities")
        total_loss = None
        for index, (image, context) in enumerate(zip(images, contexts, strict=True)):
            instruction, qformer_instruction = context
            image_hash = multimodal_content_hash({"images": [image], "videos": None})
            selected = self.fixed_store.select_groups(
                [self.sample_ids[index]],
                [self.group_ids[index]],
                [self.image_ids[index]],
                [image_hash],
            )
            lengths = selected._tensor("response_lengths").long().tolist()
            stored_ids = selected._tensor("response_token_ids").long()
            response_ids = [stored_ids[row, : int(length)] for row, length in enumerate(lengths)]
            processed = [
                self.processor.process_response_ids(
                    image,
                    instruction,
                    token_ids,
                    qformer_instruction=qformer_instruction,
                )
                for token_ids in response_ids
            ]
            batch = {key: value.to(self.model.device) for key, value in self.processor.collate(processed).items()}
            current_log_probs, current_mask = response_log_probs(
                self.model, batch, response_length=self.config.max_txt_len
            )
            actual_ids = torch.full_like(current_mask, int(self.processor.tokenizer.pad_token_id), dtype=torch.long)
            for row, token_ids in enumerate(response_ids):
                actual_ids[row, : token_ids.numel()] = token_ids.to(actual_ids.device)
            prompt_ids = self.processor.generate_inputs(
                image,
                instruction,
                qformer_instruction=qformer_instruction,
            )["input_ids"][0]
            fixed = prepare_fixed_adapter_batch(
                selected,
                prompt_token_ids=prompt_ids,
                response_token_ids=actual_ids,
                response_mask=current_mask,
                device=current_log_probs.device,
            )

            texts = [self.processor.decode(token_ids.tolist()) for token_ids in response_ids]
            answers = [self.answers[index]] * self.group_size
            tasks = [self.tasks[index]] * self.group_size
            checked_outcomes = self.outcome_backend.compute(texts, answers, tasks).to(fixed.outcome_rewards.device)
            if not torch.allclose(checked_outcomes.float(), fixed.outcome_rewards, atol=1e-6, rtol=0):
                raise ValueError("fixed InstructBLIP outcome rewards do not match the exact stored actions")

            advantages, _ = compute_fixed_copo_advantages(fixed)
            group_loss, _ = compute_copo_eq11_total_loss(
                fixed.old_log_probs,
                current_log_probs,
                fixed.reference_log_probs,
                advantages,
                fixed.response_mask,
                self.config.clip_ratio,
                self.config.clip_ratio,
                self.config.kl_loss_coef,
            )
            total_loss = group_loss if total_loss is None else total_loss + group_loss

        if total_loss is None:
            raise ValueError("fixed InstructBLIP update received no samples")
        loss = total_loss / len(images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()


def build_config(args: argparse.Namespace) -> InstructBlipConfig:
    return InstructBlipConfig(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.group_size,
        num_steps=args.steps,
        max_txt_len=args.max_new_tokens,
        clip_ratio=args.clip_ratio,
        kl_loss_coef=args.kl_coef,
        generation_protocol=args.instructblip_protocol,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train COPO with the InstructBLIP backend.")
    parser.add_argument("--model", required=True, help="InstructBLIP HF checkpoint directory")
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
    parser.add_argument("--group-size", type=int, default=8, help="responses sampled per prompt")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--steps", type=int, required=True, help="fixed-trajectory update iterations")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="stored response length limit")
    parser.add_argument(
        "--outcome-reward",
        default="answer",
        choices=["answer"],
        help="ground-truth task outcome reward required by fixed COPO training",
    )
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto", help="auto -> cuda if available else cpu")
    parser.add_argument(
        "--instructblip-protocol",
        default=ORIGINAL_PROTOCOL,
        choices=SUPPORTED_PROTOCOLS,
        help=(
            "generation/scoring context protocol; default 'original' preserves the paper/legacy "
            "prompt, while 'split_context_two_shot' is an explicit no-training adaptation"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/instructblip", help="where to save trainable weights")
    parser.add_argument("--resume-from", default=None, help="resume trainable parameters, optimizer, and step")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = build_config(args)
    config.validate()

    model = load_model(config)
    model = model.to(config.resolve_device())
    apply_freeze(model, config)

    processor = InstructBlipProcessor(config)
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

    eos_ids = generation_eos_token_ids(model, processor.tokenizer)
    if not eos_ids or processor.tokenizer.pad_token_id is None:
        raise ValueError("fixed InstructBLIP training requires explicit EOS and padding token IDs")
    expected_metadata = {
            "model_family": "instructblip",
            "tokenizer_fingerprint": fingerprint_tokenizer(processor.tokenizer),
            "processor_template_fingerprint": protocol_processor_fingerprint(
                processor._hf, config.generation_protocol
            ),
            "model_config_fingerprint": fingerprint_model_config(config.model_name_or_path),
            "group_size": config.batch_size,
            "max_response_length": config.max_txt_len,
            "lambda_s": 0.35,
            "lambda_n": 0.35,
            "eos_token_ids": [int(token) for token in eos_ids],
            "pad_token_id": int(processor.tokenizer.pad_token_id),
    }
    if not args.resume_from:
        expected_metadata["collector_policy_id"] = model_state_fingerprint(model)
    fixed_store = FixedTrajectoryStore.load(
        args.causal_reward,
        expected_metadata=expected_metadata,
    )
    validate_store_protocol(fixed_store, config.generation_protocol)
    trainer = InstructBlipCOPOTrainer(
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
            expected_model_family="instructblip",
        )
        if start_step >= config.num_steps:
            raise ValueError(
                f"resume step {start_step} is not smaller than configured steps {config.num_steps}"
            )

    logger.info("starting COPO training: %d samples, group_size=%d, steps=%d",
                len(samples), config.batch_size, config.num_steps)
    for step in range(start_step, config.num_steps):
        loss = trainer.step(images, instructions)
        logger.info("step %d/%d  loss=%.4f", step + 1, config.num_steps, loss)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    torch.save(trainable, output_dir / "instructblip_trainable.pt")
    save_trainable_checkpoint(
        output_dir / "instructblip_training_state.pt",
        model,
        trainer.optimizer,
        step=config.num_steps,
        model_family="instructblip",
        config={
            "group_size": config.batch_size,
            "max_response_length": config.max_txt_len,
            "lr": config.lr,
            "clip_ratio": config.clip_ratio,
            "kl_loss_coef": config.kl_loss_coef,
            "generation_protocol": config.generation_protocol,
        },
    )
    logger.info("saved trainable weights to %s", output_dir / "instructblip_trainable.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
