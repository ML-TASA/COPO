"""Model-agnostic collection of strict ``copo.fixed_trajectory.v3`` stores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch

from verl.utils.causal_reward.reward_manager import necessity_score, sufficiency_score
from verl.utils.causal_reward.trajectory import (
    CONTRACT_NAME,
    FORMAT_VERSION,
    POLICY_LOGPROB_SEMANTICS,
    FixedTrajectoryStore,
    make_trajectory_id,
    token_sequence_hash,
)
from verl.utils.reward_score.outcome import compute_score


@dataclass(frozen=True)
class GeneratedAction:
    token_ids: tuple[int, ...]
    stop_reason: str


@dataclass(frozen=True)
class CollectorIdentity:
    model_family: str
    tokenizer_fingerprint: str
    processor_template_fingerprint: str
    model_config_fingerprint: str
    policy_id: str
    eos_token_ids: tuple[int, ...]
    pad_token_id: int


class CollectorBackend(Protocol):
    identity: CollectorIdentity

    def prepare_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def generate_actions(
        self,
        prepared: Mapping[str, Any],
        *,
        num_samples: int,
        max_new_tokens: int,
        prefix_ids: Sequence[int] = (),
        zero_mask_index: int | None = None,
    ) -> list[GeneratedAction]: ...

    def action_log_probs(
        self, prepared: Mapping[str, Any], action_ids: Sequence[int]
    ) -> torch.Tensor: ...

    def decode_action(self, action_ids: Sequence[int]) -> str: ...


def model_state_fingerprint(model) -> str:
    """Hash the actual policy state, including adapters/projectors/LoRA."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _find(values: Sequence[int], pattern: Sequence[int], start: int = 0) -> int | None:
    pattern = tuple(int(value) for value in pattern)
    if not pattern:
        return None
    values = tuple(int(value) for value in values)
    for index in range(start, len(values) - len(pattern) + 1):
        if values[index:index + len(pattern)] == pattern:
            return index
    return None


def structured_response_masks(
    action_ids: Sequence[int],
    *,
    think_start_ids: Sequence[int],
    think_end_ids: Sequence[int],
    answer_start_ids: Sequence[int],
    answer_end_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify token spans without character-to-token index conversion."""
    ids = tuple(int(value) for value in action_ids)
    ts = _find(ids, think_start_ids)
    te = _find(ids, think_end_ids, 0 if ts is None else ts + len(think_start_ids))
    ans = _find(ids, answer_start_ids, 0 if te is None else te + len(think_end_ids))
    ane = _find(ids, answer_end_ids, 0 if ans is None else ans + len(answer_start_ids))
    if None in (ts, te, ans, ane):
        raise ValueError("generated action lacks a complete <think>/<answer> token structure")
    reasoning_start = int(ts) + len(tuple(think_start_ids))
    reasoning_stop = int(te)
    answer_start = int(ans) + len(tuple(answer_start_ids))
    answer_stop = int(ane)
    if reasoning_start >= reasoning_stop or answer_start >= answer_stop:
        raise ValueError("generated action has an empty reasoning or answer span")
    reasoning = torch.zeros(len(ids), dtype=torch.bool)
    answer = torch.zeros(len(ids), dtype=torch.bool)
    reasoning[reasoning_start:reasoning_stop] = True
    answer[answer_start:answer_stop] = True
    special = ~(reasoning | answer)
    return reasoning, answer, special


def structured_response_masks_from_decode(
    action_ids: Sequence[int],
    decode_tokens,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify structured spans without retokenizing the sampled action.

    Byte/BPE tokenizers may merge delimiter characters with adjacent content,
    so an isolated ``</think>`` encoding need not occur as an ID subsequence.
    Prefix decoding preserves the original action IDs and lets us conservatively
    exclude every token that overlaps a delimiter.
    """
    ids = tuple(int(value) for value in action_ids)
    full_text = str(decode_tokens(ids))
    markers = ("<think>", "</think>", "<answer>", "</answer>")
    ts = full_text.find(markers[0])
    te = full_text.find(markers[1], ts + len(markers[0])) if ts >= 0 else -1
    ans = full_text.find(markers[2], te + len(markers[1])) if te >= 0 else -1
    ane = full_text.find(markers[3], ans + len(markers[2])) if ans >= 0 else -1
    if min(ts, te, ans, ane) < 0:
        raise ValueError("generated action lacks a complete <think>/<answer> token structure")

    boundaries = [0]
    previous = ""
    for stop in range(1, len(ids) + 1):
        prefix = str(decode_tokens(ids[:stop]))
        if not prefix.startswith(previous) or not full_text.startswith(prefix):
            raise ValueError("tokenizer prefix decoding is not stable enough for structural alignment")
        boundaries.append(len(prefix))
        previous = prefix
    if previous != full_text:
        raise ValueError("tokenizer prefix decoding did not reproduce the sampled action")

    reasoning_chars = (ts + len(markers[0]), te)
    answer_chars = (ans + len(markers[2]), ane)
    reasoning = torch.zeros(len(ids), dtype=torch.bool)
    answer = torch.zeros(len(ids), dtype=torch.bool)
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        if start < stop and reasoning_chars[0] <= start and stop <= reasoning_chars[1]:
            reasoning[index] = True
        elif start < stop and answer_chars[0] <= start and stop <= answer_chars[1]:
            answer[index] = True
    if not reasoning.any() or not answer.any():
        raise ValueError("generated action has an empty token-aligned reasoning or answer span")
    return reasoning, answer, ~(reasoning | answer)


def prefilled_structured_response_masks(
    action_ids: Sequence[int],
    *,
    think_end_ids: Sequence[int],
    answer_start_ids: Sequence[int],
    answer_end_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify an action when the opening ``<think>`` is part of the prompt."""
    ids = tuple(int(value) for value in action_ids)
    te = _find(ids, think_end_ids)
    ans = _find(ids, answer_start_ids, 0 if te is None else te + len(think_end_ids))
    ane = _find(ids, answer_end_ids, 0 if ans is None else ans + len(answer_start_ids))
    if None in (te, ans, ane):
        raise ValueError("generated action lacks a complete prefilled <think>/<answer> token structure")
    reasoning_stop = int(te)
    answer_start = int(ans) + len(tuple(answer_start_ids))
    answer_stop = int(ane)
    if reasoning_stop <= 0 or answer_start >= answer_stop:
        raise ValueError("generated action has an empty reasoning or answer span")
    reasoning = torch.zeros(len(ids), dtype=torch.bool)
    answer = torch.zeros(len(ids), dtype=torch.bool)
    reasoning[:reasoning_stop] = True
    answer[answer_start:answer_stop] = True
    return reasoning, answer, ~(reasoning | answer)


def prefilled_structured_response_masks_from_decode(
    action_ids: Sequence[int],
    decode_tokens,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode-boundary fallback for a prompt-prefilled ``<think>`` marker."""
    ids = tuple(int(value) for value in action_ids)
    full_text = str(decode_tokens(ids))
    te = full_text.find("</think>")
    ans = full_text.find("<answer>", te + len("</think>")) if te >= 0 else -1
    ane = full_text.find("</answer>", ans + len("<answer>")) if ans >= 0 else -1
    if min(te, ans, ane) < 0:
        raise ValueError("generated action lacks a complete prefilled <think>/<answer> token structure")

    boundaries = [0]
    previous = ""
    for stop in range(1, len(ids) + 1):
        prefix = str(decode_tokens(ids[:stop]))
        if not prefix.startswith(previous) or not full_text.startswith(prefix):
            raise ValueError("tokenizer prefix decoding is not stable enough for structural alignment")
        boundaries.append(len(prefix))
        previous = prefix
    if previous != full_text:
        raise ValueError("tokenizer prefix decoding did not reproduce the sampled action")

    reasoning_chars = (0, te)
    answer_chars = (ans + len("<answer>"), ane)
    reasoning = torch.zeros(len(ids), dtype=torch.bool)
    answer = torch.zeros(len(ids), dtype=torch.bool)
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        if start < stop and reasoning_chars[0] <= start and stop <= reasoning_chars[1]:
            reasoning[index] = True
        elif start < stop and answer_chars[0] <= start and stop <= answer_chars[1]:
            answer[index] = True
    if not reasoning.any() or not answer.any():
        raise ValueError("generated action has an empty token-aligned reasoning or answer span")
    return reasoning, answer, ~(reasoning | answer)


@dataclass
class FixedTrajectoryCollector:
    collector: CollectorBackend
    reference: CollectorBackend
    group_size: int
    max_response_length: int
    num_suffix_samples: int = 8
    collection_round: int = 0
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = -1
    do_sample: bool = True
    answer_normalization: str = "strict"
    max_causal_tokens: int | None = None
    max_generation_attempts: int = 1

    def __post_init__(self):
        if self.group_size < 2 or self.num_suffix_samples < 1:
            raise ValueError("collector requires group_size >= 2 and H >= 1")
        if self.max_generation_attempts < 1:
            raise ValueError("collector max_generation_attempts must be positive")
        if (self.temperature, self.top_p, self.top_k, self.do_sample) != (1.0, 0.9, -1, True):
            raise ValueError(
                "collector backends currently implement sampling with temperature=1.0, "
                "top_p=0.9, top_k=-1, do_sample=True; metadata must describe that distribution"
            )
        if self.collector.identity.model_family != self.reference.identity.model_family:
            raise ValueError("collector and reference model families differ")
        for field in ("tokenizer_fingerprint", "processor_template_fingerprint", "model_config_fingerprint"):
            if getattr(self.collector.identity, field) != getattr(self.reference.identity, field):
                raise ValueError(f"collector/reference {field} mismatch")

    def collect(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collect an empty trajectory dataset")
        width = self.max_response_length
        rows: list[dict[str, Any]] = []
        first_prepared = self.collector.prepare_sample(samples[0])
        tokenize_special = first_prepared["tokenize_special"]
        delimiters = {
            "think_start_ids": tokenize_special("<think>"),
            "think_end_ids": tokenize_special("</think>"),
            "answer_start_ids": tokenize_special("<answer>"),
            "answer_end_ids": tokenize_special("</answer>"),
        }
        if any(not value for value in delimiters.values()):
            raise ValueError("tokenizer produced an empty structural delimiter")

        for sample_index, sample in enumerate(samples):
            for field in ("sample_id", "group_id", "image_id", "answer"):
                if not str(sample.get(field, "")).strip():
                    raise ValueError(f"collector sample is missing required field {field!r}")
            prepared = first_prepared if sample_index == 0 else self.collector.prepare_sample(sample)
            reasoning_prefill = bool(prepared.get("reasoning_prefill", False))
            accepted = []
            invalid_count = 0
            last_error = None
            for _attempt in range(self.max_generation_attempts):
                actions = self.collector.generate_actions(
                    prepared,
                    num_samples=self.group_size,
                    max_new_tokens=width,
                )
                if len(actions) != self.group_size:
                    raise ValueError("collector backend returned the wrong rollout group size")
                for action in actions:
                    ids = tuple(int(token) for token in action.token_ids)
                    if not ids or len(ids) > width:
                        invalid_count += 1
                        last_error = ValueError(
                            "generated action length is empty or exceeds max_response_length"
                        )
                        continue
                    try:
                        if reasoning_prefill:
                            masks = prefilled_structured_response_masks(
                                ids,
                                think_end_ids=delimiters["think_end_ids"],
                                answer_start_ids=delimiters["answer_start_ids"],
                                answer_end_ids=delimiters["answer_end_ids"],
                            )
                        else:
                            masks = structured_response_masks(ids, **delimiters)
                    except ValueError as token_error:
                        decode_tokens = prepared.get("decode_tokens")
                        try:
                            if not callable(decode_tokens):
                                raise token_error
                            if reasoning_prefill:
                                masks = prefilled_structured_response_masks_from_decode(ids, decode_tokens)
                            else:
                                masks = structured_response_masks_from_decode(ids, decode_tokens)
                        except ValueError as exc:
                            invalid_count += 1
                            last_error = exc
                            continue
                    accepted.append((action, ids, masks))
                    if len(accepted) == self.group_size:
                        break
                if len(accepted) == self.group_size:
                    break
            if len(accepted) != self.group_size:
                raise ValueError(
                    f"invalid structured response group for sample_id={sample['sample_id']!r}, "
                    f"group_id={sample['group_id']!r}; collected {len(accepted)}/{self.group_size} "
                    f"valid actions after {self.max_generation_attempts} bounded attempts "
                    f"({invalid_count} invalid); the collector does not modify sampled actions"
                ) from last_error

            for rollout_index, (action, ids, masks) in enumerate(accepted):
                reasoning, answer_mask, special = masks
                text = self.collector.decode_action(ids)
                task = sample.get("task")
                outcome = float(
                    compute_score(text, sample["answer"], normalization=self.answer_normalization, task=task)
                )
                suff = torch.zeros(len(ids), dtype=torch.float32)
                nec = torch.zeros(len(ids), dtype=torch.float32)
                reasoning_positions = torch.where(reasoning)[0].tolist()
                if self.max_causal_tokens is not None:
                    reasoning_positions = reasoning_positions[: self.max_causal_tokens]
                answer_delimiter_end = int(torch.where(answer_mask)[0][0])
                for position in reasoning_positions:
                    with_actions = self.collector.generate_actions(
                        prepared,
                        prefix_ids=ids[: position + 1],
                        num_samples=self.num_suffix_samples,
                        max_new_tokens=max(width - position - 1, 1),
                    )
                    without_actions = self.collector.generate_actions(
                        prepared,
                        prefix_ids=ids[:position],
                        num_samples=self.num_suffix_samples,
                        max_new_tokens=max(width - position, 1),
                    )
                    if len(with_actions) != len(without_actions) or len(with_actions) != self.num_suffix_samples:
                        raise ValueError("sufficiency continuation pairs are incomplete")
                    with_scores, without_scores = [], []
                    for with_action, without_action in zip(with_actions, without_actions, strict=True):
                        with_text = self.collector.decode_action((*ids[: position + 1], *with_action.token_ids))
                        without_text = self.collector.decode_action((*ids[:position], *without_action.token_ids))
                        with_scores.append(float(compute_score(
                            with_text, sample["answer"], normalization=self.answer_normalization, task=task
                        )))
                        without_scores.append(float(compute_score(
                            without_text, sample["answer"], normalization=self.answer_normalization, task=task
                        )))
                    suff[position] = sufficiency_score(with_scores, without_scores)

                    necessity_prefix = ids[:answer_delimiter_end]
                    masked = self.collector.generate_actions(
                        prepared,
                        prefix_ids=necessity_prefix,
                        zero_mask_index=position,
                        num_samples=1,
                        max_new_tokens=max(width - len(necessity_prefix), 1),
                    )
                    if len(masked) != 1:
                        raise ValueError("necessity intervention must return exactly one continuation")
                    masked_text = self.collector.decode_action((*necessity_prefix, *masked[0].token_ids))
                    masked_score = float(
                        compute_score(
                            masked_text, sample["answer"], normalization=self.answer_normalization, task=task
                        )
                    )
                    nec[position] = necessity_score(outcome, masked_score)

                old = self.collector.action_log_probs(prepared, ids).detach().cpu().float()
                ref_prepared = self.reference.prepare_sample(sample)
                ref = self.reference.action_log_probs(ref_prepared, ids).detach().cpu().float()
                if old.shape != (len(ids),) or ref.shape != (len(ids),):
                    raise ValueError("old/reference log-probabilities are not aligned to the generated action")
                rows.append(
                    {
                        "sample_id": str(sample["sample_id"]),
                        "group_id": str(sample["group_id"]),
                        "image_id": str(sample["image_id"]),
                        "image_hash": str(prepared["image_hash"]),
                        "prompt_ids": tuple(int(value) for value in prepared["prompt_token_ids"]),
                        "rollout_index": rollout_index,
                        "ids": ids,
                        "reasoning": reasoning,
                        "answer": answer_mask,
                        "special": special,
                        "old": old,
                        "ref": ref,
                        "outcome": outcome,
                        "suff": suff,
                        "nec": nec,
                        "stop_reason": action.stop_reason,
                        "prompt_suffix_ids": tuple(delimiters["think_start_ids"]) if reasoning_prefill else (),
                        "protocol_id": prepared.get("protocol_id"),
                    }
                )
        return self._serialize(rows)

    def _serialize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count, width = len(rows), self.max_response_length
        protocol_ids = {row.get("protocol_id") for row in rows}
        if len(protocol_ids) != 1:
            raise ValueError("fixed trajectory rows cannot mix generation protocols")
        protocol_id = next(iter(protocol_ids))
        pad = int(self.collector.identity.pad_token_id)
        prompt_width = max(len(row["prompt_ids"]) for row in rows)
        tensors = {
            "rollout_indices": torch.empty(count, dtype=torch.long),
            "prompt_token_ids": torch.full((count, prompt_width), pad, dtype=torch.long),
            "prompt_lengths": torch.empty(count, dtype=torch.long),
            "response_token_ids": torch.full((count, width), pad, dtype=torch.long),
            "response_lengths": torch.empty(count, dtype=torch.long),
            "response_mask": torch.zeros(count, width, dtype=torch.long),
            "reasoning_mask": torch.zeros(count, width, dtype=torch.long),
            "answer_mask": torch.zeros(count, width, dtype=torch.long),
            "special_token_mask": torch.zeros(count, width, dtype=torch.long),
            "terminal_token_ids": torch.empty(count, dtype=torch.long),
            "ends_with_eos": torch.empty(count, dtype=torch.bool),
            "old_policy_log_probs": torch.zeros(count, width),
            "reference_log_probs": torch.zeros(count, width),
            "outcome_rewards": torch.empty(count),
            "sufficiency": torch.zeros(count, width),
            "necessity": torch.zeros(count, width),
            "causal_reward": torch.zeros(count, width),
        }
        store: dict[str, Any] = {
            "contract": CONTRACT_NAME,
            "format_version": FORMAT_VERSION,
            "metadata": {
                "model_family": self.collector.identity.model_family,
                "tokenizer_fingerprint": self.collector.identity.tokenizer_fingerprint,
                "processor_template_fingerprint": self.collector.identity.processor_template_fingerprint,
                "collector_policy_id": self.collector.identity.policy_id,
                "reference_policy_id": self.reference.identity.policy_id,
                "collection_round": self.collection_round,
                "group_size": self.group_size,
                "model_config_fingerprint": self.collector.identity.model_config_fingerprint,
                "sampling_parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "do_sample": self.do_sample,
                    "max_new_tokens": width,
                    "num_suffix_samples": self.num_suffix_samples,
                    "max_generation_attempts": self.max_generation_attempts,
                    "prompt_suffix_token_ids": list(rows[0]["prompt_suffix_ids"]),
                },
                "answer_normalization": self.answer_normalization,
                "log_prob_semantics": POLICY_LOGPROB_SEMANTICS,
                "max_response_length": width,
                "lambda_s": 0.35,
                "lambda_n": 0.35,
                "eos_token_ids": list(self.collector.identity.eos_token_ids),
                "pad_token_id": pad,
            },
            "sample_ids": [], "group_ids": [], "trajectory_ids": [], "prompt_hashes": [],
            "image_ids": [], "image_hashes": [], "stop_reasons": [],
            "collector_policy_ids": [], "reference_policy_ids": [],
            **tensors,
        }
        if protocol_id not in (None, "original"):
            store["metadata"]["sampling_parameters"]["protocol_id"] = str(protocol_id)
        eos_ids = set(self.collector.identity.eos_token_ids)
        for index, row in enumerate(rows):
            length, prompt_length = len(row["ids"]), len(row["prompt_ids"])
            store["sample_ids"].append(row["sample_id"]); store["group_ids"].append(row["group_id"])
            store["image_ids"].append(row["image_id"]); store["image_hashes"].append(row["image_hash"])
            store["stop_reasons"].append(row["stop_reason"])
            store["collector_policy_ids"].append(self.collector.identity.policy_id)
            store["reference_policy_ids"].append(self.reference.identity.policy_id)
            prompt_hash = token_sequence_hash(row["prompt_ids"]); store["prompt_hashes"].append(prompt_hash)
            store["rollout_indices"][index] = row["rollout_index"]
            store["prompt_token_ids"][index, :prompt_length] = torch.tensor(row["prompt_ids"])
            store["prompt_lengths"][index] = prompt_length
            store["response_token_ids"][index, :length] = torch.tensor(row["ids"])
            store["response_lengths"][index] = length; store["response_mask"][index, :length] = 1
            for key, source in (("reasoning_mask", "reasoning"), ("answer_mask", "answer"), ("special_token_mask", "special")):
                store[key][index, :length] = row[source].long()
            terminal = int(row["ids"][-1]); store["terminal_token_ids"][index] = terminal
            store["ends_with_eos"][index] = terminal in eos_ids
            for key, source in (("old_policy_log_probs", "old"), ("reference_log_probs", "ref"), ("sufficiency", "suff"), ("necessity", "nec")):
                store[key][index, :length] = row[source]
            store["outcome_rewards"][index] = row["outcome"]
            store["causal_reward"][index] = 0.35 * store["sufficiency"][index] + 0.35 * store["necessity"][index]
            store["trajectory_ids"].append(make_trajectory_id(
                sample_id=row["sample_id"], group_id=row["group_id"], collection_round=self.collection_round,
                rollout_index=row["rollout_index"], prompt_hash=prompt_hash, image_hash=row["image_hash"],
                response_token_ids=list(row["ids"]), response_mask=[1] * length,
                reasoning_mask=row["reasoning"].long().tolist(), answer_mask=row["answer"].long().tolist(),
                special_token_mask=row["special"].long().tolist(), collector_policy_id=self.collector.identity.policy_id,
                reference_policy_id=self.reference.identity.policy_id, old_policy_log_probs=row["old"].tolist(),
                reference_log_probs=row["ref"].tolist(), outcome_reward=row["outcome"],
                sufficiency=row["suff"].tolist(), necessity=row["nec"].tolist(),
                causal_reward=(0.35 * row["suff"] + 0.35 * row["nec"]).tolist(),
            ))
        FixedTrajectoryStore(store)
        return store


def save_fixed_trajectory_store(path: str | Path, store: Mapping[str, Any]) -> None:
    FixedTrajectoryStore(store)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(store), path)
    sidecar = {
        "contract": store["contract"],
        "format_version": store["format_version"],
        "model_family": store["metadata"]["model_family"],
        "collection_round": store["metadata"]["collection_round"],
        "num_trajectories": len(store["sample_ids"]),
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
