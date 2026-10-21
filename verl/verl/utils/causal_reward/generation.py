from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from verl.utils.reward_score.outcome import compute_score as compute_outcome_score


@dataclass
class GenerationResult:
    generated_token_ids: list[int]
    generated_text: str
    extracted_answer: str
    success: bool
    error_reason: str | None
    sample_id: str
    mode: str
    logprobs: list[float] | None = None


def extract_answer(text, start="<answer>", end="</answer>", normalization="strict"):
    value = (text or "").strip()
    if start in value and end in value:
        value = value.split(start, 1)[1].split(end, 1)[0].strip()
    if normalization == "lenient":
        value = " ".join(value.lower().replace(".", " ").replace(",", " ").split())
    elif normalization != "strict":
        raise ValueError(f"invalid answer normalization: {normalization}")
    return value


def outcome_reward(result, label, normalization="strict", task=None):
    if label is None:
        raise ValueError("COPO outcome reward requires an answer or label")
    if not result.success:
        return 0.0
    return compute_outcome_score(
        result.extracted_answer or result.generated_text,
        label,
        normalization=normalization,
        task=task,
    )


def validate_generation_result(result):
    if not isinstance(result, GenerationResult):
        raise TypeError("generation callable must return GenerationResult or a list of GenerationResult")
    if result.success and not isinstance(result.generated_token_ids, list):
        raise ValueError("successful generation must contain generated_token_ids")
    if not result.success and not result.error_reason:
        raise ValueError("failed generation must contain error_reason")
    return result


class VerlRolloutGenerationCallable:
    def __init__(self, rollout_manager, max_new_tokens=32, normalization="strict", timeout=120.0):
        self.rollout_manager = rollout_manager
        self.max_new_tokens = max_new_tokens
        self.normalization = normalization
        self.timeout = timeout

    def __call__(self, mode, row, data, prefix=None, masked_response=None, samples=1, **kwargs):
        if mode not in {"original", "sufficiency_with", "sufficiency_without", "necessity"}:
            raise ValueError(f"invalid COPO generation mode: {mode}")
        if mode == "necessity":
            raise NotImplementedError(
                "native online rollout cannot implement COPO necessity with an embedding zero-mask; "
                "use a model-specific fixed-trajectory v3 collector"
            )
        source = data[row : row + 1]
        count = samples if mode.startswith("sufficiency") else 1
        request = source.repeat(repeat_times=count, interleave=True)
        context = masked_response if mode == "necessity" else prefix
        if context is None:
            context = torch.empty(0, dtype=data.batch["responses"].dtype)
        prefix_values = np.empty(count, dtype=object)
        prefix_values[:] = [torch.as_tensor(context).detach().cpu().tolist() for _ in range(count)]
        request.non_tensor_batch["copo_prefix_ids"] = prefix_values
        request.non_tensor_batch["copo_generation_mode"] = np.array([mode] * count, dtype=object)
        request.non_tensor_batch["copo_max_new_tokens"] = np.array([self.max_new_tokens] * count)
        request.meta_info["response_length"] = self.max_new_tokens
        started = time.perf_counter()
        try:
            output = self.rollout_manager.generate_sequences(request)
        except Exception as exc:
            sample_id = self._sample_id(data, row)
            return GenerationResult([], "", "", False, str(exc), sample_id, mode)
        elapsed = time.perf_counter() - started
        if elapsed > self.timeout:
            return GenerationResult([], "", "", False, f"generation exceeded timeout {self.timeout}", self._sample_id(data, row), mode)
        results = []
        for index in range(count):
            mask = output.batch["response_mask"][index].bool()
            token_ids = output.batch["responses"][index][mask].detach().cpu().tolist()
            texts = output.non_tensor_batch.get("generated_text")
            text = str(texts[index]) if texts is not None else ""
            logprobs = output.batch.get("rollout_log_probs")
            result = GenerationResult(
                generated_token_ids=token_ids,
                generated_text=text,
                extracted_answer=extract_answer(text, normalization=self.normalization),
                success=True,
                error_reason=None,
                sample_id=self._sample_id(data, row),
                mode=mode,
                logprobs=logprobs[index][mask].detach().cpu().tolist() if logprobs is not None else None,
            )
            results.append(validate_generation_result(result))
        return results if count > 1 else results[0]

    @staticmethod
    def _sample_id(data, row):
        values = data.non_tensor_batch.get("sample_id", data.non_tensor_batch.get("uid"))
        return str(values[row]) if values is not None else str(row)

