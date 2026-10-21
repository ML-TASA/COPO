"""COPO reward generation and strict fixed-trajectory collection.

The command-line entry point dispatches to a model-specific backend and writes
``copo.fixed_trajectory.v3``. The small ``CausalRewardGenerator`` class below
is retained for tensor-level reward tests; its v2 output is not accepted by
fixed-policy training.

Pipeline (per sample)::

    dataset sample
         |
         v
    model inference (original responses)
         |
         +----------------+
         |                |
         v                v
    sufficiency      necessity
         |
         v
    causal reward aggregation
         |
         v
    save precomputed reward store

Formal collection uses :class:`verl.utils.causal_reward.collector.FixedTrajectoryCollector`
and the four implementations in ``causal_reward.backends``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import torch

from verl.implementations.presets import MODEL_PRESETS, get_preset
from verl.implementations.instructblip.processor import ORIGINAL_PROTOCOL, SUPPORTED_PROTOCOLS
from verl.utils.causal_reward.integration import COPOConfig, build_causal_reward_tensor
from verl.utils.causal_reward.reward_manager import mask_token, necessity_score, sufficiency_score
from verl.utils.reward_score.outcome import compute_score

logger = logging.getLogger(__name__)

#: Current on-disk format of the precomputed reward store.
FORMAT_VERSION = 2

#: Tensor keys written to the ``.pt`` reward store.
STORE_TENSOR_KEYS = (
    "sufficiency",
    "necessity",
    "causal_reward",
    "reasoning_mask",
    "response_token_ids",
    "response_lengths",
)
STORE_KEYS = (*STORE_TENSOR_KEYS, "sample_ids")


def score_answer(text: str, answer: str, normalization: str, task: str | None) -> float:
    """Score a generated answer against a ground-truth answer.

    Delegates to the same outcome scorer used during COPO training so that
    offline and online rewards are identical.
    """
    return float(compute_score(text, answer, normalization=normalization, task=task))


class InferenceBackend(Protocol):
    """Minimal model interface required by the reward generator.

    Any object implementing these two methods can generate COPO causal rewards,
    which is how new models or inference engines are plugged in.
    """

    def tokenize(self, text: str) -> list[int]:
        """Encode a model response (reasoning + answer) into token ids."""
        ...

    def generate(
        self,
        sample: dict[str, Any],
        prefix_ids: Sequence[int] | None = None,
        max_new_tokens: int = 32,
        num_samples: int = 1,
    ) -> list[str]:
        """Generate ``num_samples`` completions for one sample.

        When ``prefix_ids`` is ``None``, a full response is generated from the
        image and question. When ``prefix_ids`` is provided, generation is
        continued from that token prefix (used for sufficiency and necessity).
        """
        ...


class TransformersInferenceBackend:
    """Reference backend using a Hugging Face ``AutoProcessor``/``AutoModelForCausalLM``.

    Args:
        checkpoint: Model/checkpoint path or identifier.
        device: Torch device (``"cuda"``, ``"cuda:0"``, ``"cpu"``, ...).
        batch_size: Number of samples decoded in a single ``model.generate`` call.
    """

    def __init__(self, checkpoint: str, device: str = "cuda", batch_size: int = 8):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.batch_size = batch_size
        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.model.eval()

    def _tokenizer(self):
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("processor does not expose a tokenizer; provide a custom InferenceBackend")
        return tokenizer

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer().encode(text or "", add_special_tokens=False)

    def generate(
        self,
        sample: dict[str, Any],
        prefix_ids: Sequence[int] | None = None,
        max_new_tokens: int = 32,
        num_samples: int = 1,
    ) -> list[str]:
        tokenizer = self._tokenizer()
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["prompt"]}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[prompt] * num_samples, images=[sample["image"]] * num_samples, return_tensors="pt").to(self.device)
        if prefix_ids is not None:
            prefix = torch.as_tensor(list(prefix_ids), dtype=torch.long, device=self.device).unsqueeze(0).repeat(num_samples, 1)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        sampling_kwargs = {"do_sample": False}
        if num_samples > 1:
            sampling_kwargs = {"do_sample": True, "temperature": 1.0, "top_p": 0.9}
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                **sampling_kwargs,
            )
        generated = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)


@dataclass
class CausalRewardGenerator:
    """Orchestrates offline COPO causal-reward generation.

    Args:
        backend: The model inference backend.
        config: COPO reward configuration (lambdas, masking, scope, ...).
        max_response_length: Padded response length ``T`` of the output store.
        num_rollouts: Responses generated per sample (must match the number used during training).
        max_causal_tokens: Cap on reasoning positions scored per response
            (``None`` means no cap).
        answer_normalization: Outcome-scorer normalization (``strict``/``lenient``).
        answer_start_token_ids / answer_end_token_ids: Token ids that delimit the
            answer span. When empty, ``<answer>``/``</answer>`` are tokenized and
            located automatically.
    """

    backend: InferenceBackend
    config: COPOConfig
    max_response_length: int = 8192
    num_rollouts: int = 8  # must match the training GRPO group size
    max_causal_tokens: int | None = None
    answer_normalization: str = "strict"
    answer_start_token_ids: tuple[int, ...] = ()
    answer_end_token_ids: tuple[int, ...] = ()

    def _delimiter_ids(self) -> tuple[list[int], list[int]]:
        starts = list(self.answer_start_token_ids)
        ends = list(self.answer_end_token_ids)
        if not starts:
            starts = self.backend.tokenize("<answer>")
        if not ends:
            ends = self.backend.tokenize("</answer>")
        return starts, ends

    def _reasoning_positions(self, tokens: list[int], starts: Sequence[int]) -> list[int]:
        starts = list(starts)
        for index in range(len(tokens) - len(starts) + 1):
            if tokens[index : index + len(starts)] == starts:
                return list(range(index))
        return list(range(len(tokens)))

    @staticmethod
    def _delimiter_index(tokens: Sequence[int], delimiter: Sequence[int]) -> int | None:
        delimiter = list(delimiter)
        for index in range(len(tokens) - len(delimiter) + 1):
            if list(tokens[index : index + len(delimiter)]) == delimiter:
                return index
        return None

    def generate(self, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Compute the causal-reward store for ``samples``.

        Returns a dict with ``sample_ids``, ``sufficiency``, ``necessity``,
        ``causal_reward`` and ``format_version`` keys.
        """
        samples = list(samples)
        n = self.num_rollouts
        t = self.max_response_length
        start_ids = self._delimiter_ids()[0]

        num_rows = len(samples) * n
        sufficiency = torch.zeros(num_rows, t, dtype=torch.float32)
        necessity = torch.zeros(num_rows, t, dtype=torch.float32)
        response_mask = torch.zeros(num_rows, t, dtype=torch.float32)
        reasoning_mask = torch.zeros(num_rows, t, dtype=torch.float32)
        response_token_ids = torch.zeros(num_rows, t, dtype=torch.long)
        response_lengths = torch.zeros(num_rows, dtype=torch.long)
        sample_ids: list[str] = []

        for prompt_index, sample in enumerate(samples):
            answer = sample["answer"]
            task = sample.get("task")
            responses = self.backend.generate(sample, max_new_tokens=t, num_samples=n)
            if len(responses) != n:
                raise ValueError(
                    f"inference backend returned {len(responses)} responses for sample "
                    f"{sample['sample_id']!r}; expected num_rollouts={n}"
                )
            for k, response in enumerate(responses):
                row = prompt_index * n + k
                sample_ids.append(f"{sample['sample_id']}::{k}")

                tokens = self.backend.tokenize(response)
                if len(tokens) > t:
                    raise ValueError(
                        f"generated response for sample {sample['sample_id']!r}, rollout {k} has "
                        f"{len(tokens)} tokens, exceeding max_response_length={t}; refusing to "
                        "truncate because that would break causal-reward alignment"
                    )
                reasoning_positions = self._reasoning_positions(tokens, start_ids)
                answer_delimiter = self._delimiter_index(tokens, start_ids)
                if self.max_causal_tokens is not None:
                    reasoning_positions = reasoning_positions[: self.max_causal_tokens]

                response_mask[row, : len(tokens)] = 1.0
                response_token_ids[row, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
                response_lengths[row] = len(tokens)
                for pos in reasoning_positions:
                    if pos < t:
                        reasoning_mask[row, pos] = 1.0

                original_score = score_answer(response, answer, self.answer_normalization, task)
                for pos in reasoning_positions:
                    if pos >= t:
                        break
                    if self.config.enable_sufficiency:
                        with_scores = [
                            score_answer(text, answer, self.answer_normalization, task)
                            for text in self.backend.generate(
                                sample,
                                prefix_ids=tokens[: pos + 1],
                                max_new_tokens=max(t - (pos + 1), 1),
                                num_samples=self.config.num_suffix_samples,
                            )
                        ]
                        without_scores = [
                            score_answer(text, answer, self.answer_normalization, task)
                            for text in self.backend.generate(
                                sample,
                                prefix_ids=tokens[:pos],
                                max_new_tokens=max(t - pos, 1),
                                num_samples=self.config.num_suffix_samples,
                            )
                        ]
                        sufficiency[row, pos] = sufficiency_score(with_scores, without_scores)
                    if self.config.enable_necessity:
                        # Necessity intervenes on the complete reasoning chain and
                        # then regenerates the answer.  Never append store padding
                        # or the already-generated answer as valid prefix tokens.
                        prefix_end = (
                            answer_delimiter + len(start_ids)
                            if answer_delimiter is not None
                            else len(tokens)
                        )
                        reasoning_prefix = torch.as_tensor(tokens[:prefix_end], dtype=torch.long)
                        masked = mask_token(
                            reasoning_prefix,
                            pos,
                            torch.ones_like(reasoning_prefix, dtype=torch.bool),
                            self.config.masking,
                        )
                        masked_scores = [
                            score_answer(text, answer, self.answer_normalization, task)
                            for text in self.backend.generate(
                                sample,
                                prefix_ids=masked.tolist(),
                                max_new_tokens=max(t - len(masked), 1),
                                num_samples=1,
                            )
                        ]
                        necessity[row, pos] = necessity_score(original_score, masked_scores[0])

        causal = build_causal_reward_tensor(
            response_mask,
            self.config,
            sufficiency_reward=sufficiency,
            necessity_reward=necessity,
            reasoning_mask=reasoning_mask,
        )
        return {
            "format_version": FORMAT_VERSION,
            "sample_ids": sample_ids,
            "sufficiency": sufficiency,
            "necessity": necessity,
            "causal_reward": causal,
            "reasoning_mask": reasoning_mask,
            "response_token_ids": response_token_ids,
            "response_lengths": response_lengths,
        }


def load_samples(path: str) -> list[dict[str, Any]]:
    """Load samples from a JSONL file.

    Each line is a JSON object with ``sample_id``, ``image`` (path or URL),
    ``prompt``, ``answer``, and an optional ``task`` field. The ``image`` value
    is passed through unchanged so backends may load it as needed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")
    samples = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_no} of {path}: {exc}") from exc
        for key in ("sample_id", "image", "prompt", "answer"):
            if key not in item:
                raise ValueError(f"line {line_no} of {path} is missing required field {key!r}")
        samples.append(item)
    return samples


def save_rewards(path: str, store: dict[str, Any], dataset: str | None = None) -> None:
    """Save a causal-reward store to a ``.pt`` file plus a sidecar ``.json``.

    The ``.pt`` file contains the reward tensors and the exact response token IDs
    they label. The sidecar ``.json`` records
    the format version, the per-row ``sample_ids`` ordering, the tensor shapes,
    and an optional dataset identifier so the store can be checked against the
    training samples before use.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: store[key] for key in STORE_KEYS}, path)
    shapes = {key: tuple(store[key].shape) for key in STORE_TENSOR_KEYS}
    meta = {
        "format_version": store.get("format_version"),
        "sample_ids": store.get("sample_ids"),
        "num_rows": len(store.get("sample_ids", [])),
        "tensor_shapes": shapes,
    }
    if dataset is not None:
        meta["dataset"] = Path(dataset).name
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")


def _build_config(args: argparse.Namespace) -> COPOConfig:
    return COPOConfig(
        lambda_s=args.lambda_s,
        lambda_n=args.lambda_n,
        enable_sufficiency=args.sufficiency,
        enable_necessity=args.necessity,
        num_suffix_samples=args.num_suffix_samples,
        reward_scope=args.reward_scope,
        masking=args.masking,
        advantage_mode="additive",
    )


def load_collection_samples(path: str) -> list[dict[str, Any]]:
    """Load the fail-closed schema used by the v3 trajectory collector."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {source}")
    samples = []
    for index, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {index} of {source}: {exc}") from exc
        for key in ("sample_id", "group_id", "image_id", "image", "answer", "task"):
            if not str(sample.get(key, "")).strip():
                raise ValueError(f"line {index} of {source} is missing required v3 field {key!r}")
        instruction = sample.get("instruction", sample.get("prompt"))
        if not str(instruction or "").strip():
            raise ValueError(f"line {index} of {source} has no instruction/prompt")
        if not Path(str(sample["image"])).is_file():
            raise FileNotFoundError(f"line {index} of {source} image does not exist: {sample['image']}")
        sample["instruction"] = str(instruction)
        samples.append(sample)
    if not samples:
        raise ValueError(f"dataset file contains no samples: {source}")
    return samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect strict COPO v3 fixed trajectories.")
    parser.add_argument("--model", dest="model_name", required=True, choices=sorted(MODEL_PRESETS),
                        help="Formal experiment model family.")
    parser.add_argument(
        "--model-path", required=True,
        help="Local model checkpoint; for MiniGPT-4 this is the Vicuna/LLaMA directory",
    )
    parser.add_argument(
        "--reference-model", required=True,
        help="Local frozen reference checkpoint; for MiniGPT-4 this is the reference stage-2 checkpoint",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="validated JSONL with sample_id/group_id/image_id/image/instruction/answer/task",
    )
    parser.add_argument("--output", required=True, help="Output copo.fixed_trajectory.v3 .pt path")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda, cuda:0, cpu, ...)")
    parser.add_argument(
        "--dtype", default=None, choices=["float16", "bfloat16", "float32"],
        help="InstructBLIP/MiniGPT-4 model dtype; defaults to the backend's established setting",
    )
    parser.add_argument(
        "--instructblip-protocol",
        default=ORIGINAL_PROTOCOL,
        choices=SUPPORTED_PROTOCOLS,
        help=(
            "InstructBLIP-only generation context; default 'original' preserves the formal "
            "single-stage prefilled protocol"
        ),
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=None,
        help="Override the preset GRPO group size; training must use the same value",
    )
    parser.add_argument(
        "--max-response-length", type=int, default=None,
        help="Override the preset response cap; training must use the same value",
    )
    parser.add_argument("--max-causal-tokens", type=int, default=None, help="Cap reasoning positions per response")
    parser.add_argument("--answer-normalization", default="strict", choices=["strict", "lenient"])
    parser.add_argument("--num-suffix-samples", type=int, default=8)
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=1,
        help="Bounded attempts to obtain a complete structured rollout group; sampled actions are never repaired",
    )
    parser.add_argument("--collection-round", type=int, default=0)
    parser.add_argument(
        "--collector-state", default=None,
        help="Prior trainable state: PEFT adapter directory for native models, training-state file for adapters",
    )
    parser.add_argument("--llama-model", default=None, help="MiniGPT-4 Vicuna/LLaMA checkpoint")
    parser.add_argument(
        "--ckpt", default=None,
        help="MiniGPT-4 collector stage-2 checkpoint matching the selected 7B/13B language-model base",
    )
    parser.add_argument("--vit-pretrained-path", default=None, help="MiniGPT-4 vision checkpoint")
    parser.add_argument("--qformer-model", default=None, help="MiniGPT-4 Q-Former checkpoint")
    parser.add_argument("--bert-model-path", default=None, help="MiniGPT-4 local BERT snapshot")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preset = get_preset(args.model_name)
    logging.basicConfig(level=logging.INFO)
    if args.model_name != "instructblip" and args.instructblip_protocol != ORIGINAL_PROTOCOL:
        raise ValueError("--instructblip-protocol is only valid with --model instructblip")
    num_rollouts = args.num_rollouts if args.num_rollouts is not None else preset["group_size"]
    max_response_length = args.max_response_length if args.max_response_length is not None else preset["max_response_length"]
    if num_rollouts < 2:
        raise ValueError("--num-rollouts must be at least 2 for GRPO")
    if max_response_length < 1:
        raise ValueError("--max-response-length must be positive")
    from verl.utils.causal_reward.backends import build_backend
    from verl.utils.causal_reward.collector import FixedTrajectoryCollector, save_fixed_trajectory_store

    samples = load_collection_samples(args.dataset)
    backend_kwargs = {"device": args.device, "max_response_length": max_response_length}
    if args.dtype:
        backend_kwargs["dtype"] = args.dtype
    if args.model_name == "instructblip":
        backend_kwargs["generation_protocol"] = args.instructblip_protocol
    collector_checkpoint = args.model_path
    if args.model_name == "minigpt4":
        required = {
            "llama_model": args.llama_model or args.model_path,
            "vit_pretrained_path": args.vit_pretrained_path,
            "qformer_model": args.qformer_model,
            "bert_model_path": args.bert_model_path,
            "ckpt": args.ckpt,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise ValueError(f"MiniGPT-4 collector requires: {flags}")
        collector_checkpoint = required.pop("ckpt")
        backend_kwargs.update(required)

    required_paths = {
        "model checkpoint": args.model_path,
        "collector checkpoint": collector_checkpoint,
        "reference checkpoint": args.reference_model,
    }
    if args.model_name == "minigpt4":
        required_paths.update({
            "vision checkpoint": args.vit_pretrained_path,
            "Q-Former checkpoint": args.qformer_model,
            "BERT snapshot": args.bert_model_path,
        })
    missing_paths = [f"{label}: {value}" for label, value in required_paths.items() if not Path(str(value)).exists()]
    if missing_paths:
        raise FileNotFoundError("required local resources do not exist: " + "; ".join(missing_paths))

    if args.collector_state and not Path(args.collector_state).exists():
        raise FileNotFoundError(f"collector trainable state does not exist: {args.collector_state}")
    collector_kwargs = dict(backend_kwargs)
    if args.collector_state:
        collector_kwargs["trainable_state"] = args.collector_state
    collector_backend = build_backend(args.model_name, checkpoint=collector_checkpoint, **collector_kwargs)
    reference_backend = build_backend(args.model_name, checkpoint=args.reference_model, **backend_kwargs)
    collector = FixedTrajectoryCollector(
        collector=collector_backend,
        reference=reference_backend,
        group_size=num_rollouts,
        max_response_length=max_response_length,
        num_suffix_samples=args.num_suffix_samples,
        collection_round=args.collection_round,
        max_causal_tokens=args.max_causal_tokens,
        answer_normalization=args.answer_normalization,
        max_generation_attempts=args.max_generation_attempts,
    )
    store = collector.collect(samples)
    save_fixed_trajectory_store(args.output, store)
    logger.info("wrote %s for %d samples -> %s", store["contract"], len(samples), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
