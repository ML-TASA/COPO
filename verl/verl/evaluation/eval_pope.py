import argparse
import json
import os
import re
import sys
from pathlib import Path

# POPE evaluation for COPO: vLLM inference and direct metric computation.
_VLLM_AVAILABLE = True
try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - depends on optional vLLM install
    _VLLM_AVAILABLE = False

_DATASETS_AVAILABLE = True
try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    _DATASETS_AVAILABLE = False

_PROCESSOR_AVAILABLE = True
try:
    from transformers import AutoProcessor
except Exception:  # pragma: no cover
    _PROCESSOR_AVAILABLE = False


VALID_SPLITS = ("default", "random", "popular", "adversarial")
SYSTEM_PROMPT = (
    "You are a helpful visual assistant. Answer the user's question about the "
    "image with a single short answer. You MUST answer only 'Yes' or 'No'."
)


def _error(message):
    print(f"[POPE ERROR] {message}", file=sys.stderr)
    raise SystemExit(2)


def normalize_answer(text):
    """Return 'yes', 'no', or None for a raw model output.

    Accepts answers enclosed in <answer>...</answer> tags (preferred) and, as
    a fallback, the first standalone yes/no token in the text.
    """
    value = (text or "").strip()
    if not value:
        return None
    lowered = value.lower()
    match = re.search(r"<answer\s*>(.*?)</answer\s*>", lowered, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r"[^a-z]", "", candidate)
        if candidate in ("yes", "no"):
            return candidate
        if candidate.startswith("yes"):
            return "yes"
        if candidate.startswith("no"):
            return "no"
        return None
    tokens = re.findall(r"\b(yes|no)\b", lowered)
    if not tokens:
        return None
    first = tokens[0].lower()
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return None


def _normalize_ground_truth(answer):
    value = (answer or "").strip().lower()
    value = re.sub(r"[^a-z]", "", value)
    if value.startswith("yes"):
        return "yes"
    if value.startswith("no"):
        return "no"
    return None


def compute_metrics(ground_truths, predictions):
    """Compute accuracy/precision/recall/F1 treating 'yes' as positive."""
    total = len(ground_truths)
    if total != len(predictions):
        raise ValueError(
            f"ground_truths and predictions length mismatch: {len(ground_truths)} vs {len(predictions)}"
        )
    tp = fp = tn = fn = 0
    for gt, pred in zip(ground_truths, predictions):
        if pred == "yes" and gt == "yes":
            tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
        elif pred == "no" and gt == "no":
            tn += 1
        elif pred == "no" and gt == "yes":
            fn += 1
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "total": total,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def validate_checkpoint(path):
    path = Path(path)
    if not path.exists():
        _error(f"checkpoint path does not exist: {path}")
    if not path.is_dir():
        _error(f"checkpoint path is not a directory: {path}")
    if not (path / "config.json").is_file():
        _error(f"checkpoint directory is missing config.json: {path}")
    return str(path.resolve())


def validate_lora_checkpoint(path):
    """Validate a standard PEFT adapter directory."""
    path = Path(path)
    if not path.exists():
        _error(f"LoRA adapter path does not exist: {path}")
    if not path.is_dir():
        _error(f"LoRA adapter path is not a directory: {path}")
    if not (path / "adapter_config.json").is_file():
        _error(f"LoRA adapter directory is missing adapter_config.json: {path}")
    return str(path.resolve())


def _generated_continuation(output_ids, input_length, is_encoder_decoder=False):
    """Return generated tokens only, excluding a decoder-only prompt prefix."""
    if is_encoder_decoder:
        return output_ids
    return output_ids[input_length:]


def _select_transformers_model_class(config):
    """Choose the matching Transformers auto class for a loaded config.

    LLaVA and InstructBLIP are image-to-text models in current Transformers;
    trying to load them through ``AutoModelForCausalLM`` fails before
    inference.  Legacy remote-code models such as Qwen-VL still use the causal
    LM auto class, so the choice must be config driven rather than global.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    if type(config) in AutoModelForImageTextToText._model_mapping:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_dataset_records(split):
    if not _DATASETS_AVAILABLE:
        _error("the 'datasets' package is required for POPE evaluation")
    if split not in VALID_SPLITS:
        _error(f"invalid split '{split}'; expected one of {VALID_SPLITS}")
    dataset = load_dataset("lmms-lab/POPE", split, split="test")
    records = []
    seen_ids = set()
    for index, item in enumerate(dataset):
        question = item.get("question")
        answer = item.get("answer")
        image = item.get("image")
        if question is None or answer is None or image is None:
            raise ValueError(
                f"POPE sample {index} is missing required fields (question/answer/image)"
            )
        question_id = item.get("question_id", index)
        if question_id in seen_ids:
            print(f"[POPE WARN] duplicate question_id={question_id}; keeping first occurrence")
            continue
        seen_ids.add(question_id)
        records.append(
            {
                "question_id": question_id,
                "question": str(question).strip(),
                "answer": answer,
                "image": image.convert("RGB"),
                "category": item.get("category", split),
            }
        )
    if not records:
        _error(f"POPE split '{split}' produced no samples")
    return records


def build_prompts(records, processor):
    prompts = []
    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": record["question"]}]},
        ]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(
            {"prompt": rendered, "multi_modal_data": {"image": record["image"]}}
        )
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a COPO checkpoint on POPE.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained HF model checkpoint")
    parser.add_argument("--lora-path", type=str, default=None, help="Optional path to a LoRA adapter directory")
    parser.add_argument("--split", type=str, default="adversarial", choices=VALID_SPLITS, help="POPE split to evaluate")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max tokens per response")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory fraction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling")
    parser.add_argument("--output", type=str, default=None, help="Path to write the JSON results file")
    parser.add_argument("--no-vllm", action="store_true", help="Use transformers generation instead of vLLM")
    return parser.parse_args()


def _transformers_inference(checkpoint, lora_path, prompts, max_new_tokens, seed):
    import torch
    from transformers import AutoConfig, AutoProcessor

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    model_class = _select_transformers_model_class(config)
    model = model_class.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    predictions = []
    for entry in prompts:
        inputs = processor(
            text=entry["prompt"], images=entry["multi_modal_data"]["image"], return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = _generated_continuation(
            output[0],
            inputs["input_ids"].shape[1],
            is_encoder_decoder=bool(getattr(model.config, "is_encoder_decoder", False)),
        )
        text = processor.decode(generated_ids, skip_special_tokens=True)
        answer = normalize_answer(text)
        predictions.append(answer)
    return predictions


def main():
    args = parse_args()
    if not _VLLM_AVAILABLE and not args.no_vllm:
        _error("vLLM is not importable; reinstall vllm or pass --no-vllm")
    if not _PROCESSOR_AVAILABLE:
        _error("the 'transformers' package is required for POPE evaluation")
    if args.batch_size <= 0:
        _error("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        _error("--max-new-tokens must be positive")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        _error("--gpu-memory-utilization must be in (0, 1]")
    checkpoint = validate_checkpoint(args.checkpoint)
    if args.lora_path is not None:
        args.lora_path = validate_lora_checkpoint(args.lora_path)

    print("=" * 60)
    print("COPO POPE Hallucination Evaluation")
    print(f"Checkpoint: {checkpoint}")
    print(f"LoRA path : {args.lora_path or '(none)'}")
    print(f"Split     : {args.split}")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    records = load_dataset_records(args.split)
    ground_truths = []
    prompts = []
    for record in records:
        gt = _normalize_ground_truth(record["answer"])
        if gt is None:
            print(f"[POPE WARN] sample {record['question_id']} has unparseable ground truth; skipping")
            continue
        ground_truths.append(gt)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": record["question"]}]},
        ]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append({"prompt": rendered, "multi_modal_data": {"image": record["image"]}})

    print(f"Evaluating {len(prompts)} samples on split '{args.split}'...")

    if args.no_vllm:
        predictions = _transformers_inference(
            checkpoint, args.lora_path, prompts, args.max_new_tokens, args.seed
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed
        )
        llm_kwargs = dict(
            model=checkpoint,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        )
        if args.lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            llm_kwargs["max_lora_rank"] = 64
        llm = LLM(**llm_kwargs)
        if args.lora_path:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("pope", 1, args.lora_path)
            outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=True)
        else:
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        predictions = []
        for out in outputs:
            text = out.outputs[0].text
            answer = normalize_answer(text)
            predictions.append(answer)

    if len(predictions) != len(ground_truths):
        _error(
            f"prediction/ground-truth count mismatch: {len(predictions)} vs {len(ground_truths)}"
        )

    malformed = sum(1 for pred in predictions if pred is None)
    if malformed:
        print(f"[POPE WARN] {malformed} unparseable responses treated as 'no'")

    predictions = [pred if pred is not None else "no" for pred in predictions]

    metrics = compute_metrics(ground_truths, predictions)
    report = {
        "checkpoint": checkpoint,
        "lora_path": args.lora_path,
        "split": args.split,
        "seed": args.seed,
        "malformed": malformed,
        "metrics": metrics,
    }

    print("=" * 60)
    print("POPE Evaluation Results")
    print("=" * 60)
    print(f"Total    : {metrics['total']}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
