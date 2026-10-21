"""Strict fixed-trajectory data contract for offline COPO optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CONTRACT_NAME = "copo.fixed_trajectory.v3"
FORMAT_VERSION = 3
POLICY_LOGPROB_SEMANTICS = "untempered_model_policy"

REQUIRED_METADATA = (
    "model_family",
    "tokenizer_fingerprint",
    "processor_template_fingerprint",
    "collector_policy_id",
    "reference_policy_id",
    "collection_round",
    "group_size",
    "model_config_fingerprint",
    "sampling_parameters",
    "log_prob_semantics",
    "max_response_length",
    "lambda_s",
    "lambda_n",
    "eos_token_ids",
    "pad_token_id",
)

ROW_LIST_FIELDS = (
    "sample_ids",
    "group_ids",
    "trajectory_ids",
    "prompt_hashes",
    "image_ids",
    "image_hashes",
    "stop_reasons",
    "collector_policy_ids",
    "reference_policy_ids",
)

ROW_TENSOR_FIELDS = (
    "rollout_indices",
    "prompt_token_ids",
    "prompt_lengths",
    "response_token_ids",
    "response_lengths",
    "response_mask",
    "reasoning_mask",
    "answer_mask",
    "special_token_mask",
    "terminal_token_ids",
    "ends_with_eos",
    "old_policy_log_probs",
    "reference_log_probs",
    "outcome_rewards",
    "sufficiency",
    "necessity",
    "causal_reward",
)

TOKEN_FIELDS = (
    "response_token_ids",
    "response_mask",
    "reasoning_mask",
    "answer_mask",
    "special_token_mask",
    "old_policy_log_probs",
    "reference_log_probs",
    "sufficiency",
    "necessity",
    "causal_reward",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_sequence_hash(token_ids: Sequence[int]) -> str:
    return _canonical_hash([int(token) for token in token_ids])


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    """Fingerprint a portable model/processor configuration mapping."""
    return _canonical_hash(value)


def multimodal_content_hash(value: Any) -> str:
    """Hash loaded multimodal content without serializing local source paths."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps(item, sort_keys=True).encode("utf-8"))
        elif isinstance(item, bytes):
            digest.update(b"bytes:")
            digest.update(item)
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(f"ndarray:{array.dtype}:{array.shape}:".encode("utf-8"))
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}:".encode("utf-8"))
            for value_item in item:
                update(value_item)
        elif all(hasattr(item, attribute) for attribute in ("mode", "size", "tobytes")):
            digest.update(f"image:{item.mode}:{tuple(item.size)}:".encode("utf-8"))
            digest.update(item.tobytes())
        else:
            raise TypeError(f"unsupported multimodal value for content fingerprint: {type(item).__qualname__}")

    update(value)
    return digest.hexdigest()


def fingerprint_tokenizer(tokenizer) -> str:
    if tokenizer is None:
        raise ValueError("fixed trajectories require a tokenizer")
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab": sorted((str(token), int(index)) for token, index in vocab.items()),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    return _canonical_hash(payload)


def fingerprint_processor_template(processor) -> str:
    if processor is None:
        raise ValueError("fixed multimodal trajectories require a processor")
    image_processor = getattr(processor, "image_processor", None)
    payload = {
        "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "chat_template": getattr(processor, "chat_template", None),
        "processor": processor.to_dict() if hasattr(processor, "to_dict") else {},
        "image_processor": image_processor.to_dict() if hasattr(image_processor, "to_dict") else {},
    }
    return _canonical_hash(payload)


def fingerprint_model_config(model_path: str | Path) -> str:
    path = Path(model_path)
    config = path / "config.json" if path.is_dir() else path
    if not config.is_file():
        raise ValueError(f"cannot fingerprint model config: {config} does not exist")
    try:
        value = json.loads(config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot fingerprint model config {config}: {exc}") from exc
    return _canonical_hash(value)


def make_trajectory_id(
    *,
    sample_id: str,
    group_id: str,
    collection_round: int,
    rollout_index: int,
    prompt_hash: str,
    image_hash: str,
    response_token_ids: Sequence[int],
    response_mask: Sequence[int],
    reasoning_mask: Sequence[int],
    answer_mask: Sequence[int],
    special_token_mask: Sequence[int],
    collector_policy_id: str,
    reference_policy_id: str,
    old_policy_log_probs: Sequence[float],
    reference_log_probs: Sequence[float],
    outcome_reward: float,
    sufficiency: Sequence[float],
    necessity: Sequence[float],
    causal_reward: Sequence[float],
) -> str:
    return _canonical_hash(
        {
            "sample_id": str(sample_id),
            "group_id": str(group_id),
            "collection_round": int(collection_round),
            "rollout_index": int(rollout_index),
            "prompt_hash": str(prompt_hash),
            "image_hash": str(image_hash),
            "response_token_ids": [int(token) for token in response_token_ids],
            "response_mask": [int(value) for value in response_mask],
            "reasoning_mask": [int(value) for value in reasoning_mask],
            "answer_mask": [int(value) for value in answer_mask],
            "special_token_mask": [int(value) for value in special_token_mask],
            "collector_policy_id": str(collector_policy_id),
            "reference_policy_id": str(reference_policy_id),
            "old_policy_log_probs": [float(value) for value in old_policy_log_probs],
            "reference_log_probs": [float(value) for value in reference_log_probs],
            "outcome_reward": float(outcome_reward),
            "sufficiency": [float(value) for value in sufficiency],
            "necessity": [float(value) for value in necessity],
            "causal_reward": [float(value) for value in causal_reward],
        }
    )


def _require_sha256(value: object, name: str, row: int | None = None) -> None:
    text = str(value)
    suffix = "" if row is None else f" at row {row}"
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ValueError(f"{name}{suffix} must be a SHA-256 hex digest")


def _require_portable_identifier(value: object, name: str, row: int | None = None) -> None:
    text = str(value).strip()
    suffix = "" if row is None else f" at row {row}"
    if not text:
        raise ValueError(f"{name}{suffix} must be nonempty")
    if text.startswith(("/", "~", "file://")) or (len(text) > 2 and text[1:3] in {":/", ":\\"}):
        raise ValueError(f"{name}{suffix} must not contain a local absolute path")


def _validate_metadata(metadata: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"fixed trajectory metadata is missing required fields: {missing}")
    if int(metadata["group_size"]) < 2:
        raise ValueError("fixed trajectory group_size must be at least 2")
    if int(metadata["max_response_length"]) < 1:
        raise ValueError("fixed trajectory max_response_length must be positive")
    if int(metadata["collection_round"]) < 0:
        raise ValueError("fixed trajectory collection_round must be non-negative")
    if not str(metadata["model_family"]).strip():
        raise ValueError("fixed trajectory model_family must be nonempty")
    if metadata["log_prob_semantics"] != POLICY_LOGPROB_SEMANTICS:
        raise ValueError(
            "fixed trajectory log-probabilities must use untempered model-policy semantics; "
            f"got {metadata['log_prob_semantics']!r}"
        )
    if float(metadata["lambda_s"]) != 0.35 or float(metadata["lambda_n"]) != 0.35:
        raise ValueError("fixed trajectory metadata must use lambda_s=lambda_n=0.35")
    if not isinstance(metadata["sampling_parameters"], Mapping):
        raise ValueError("fixed trajectory sampling_parameters must be a mapping")
    prompt_suffix = metadata["sampling_parameters"].get("prompt_suffix_token_ids", [])
    if not isinstance(prompt_suffix, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in prompt_suffix
    ):
        raise ValueError("sampling_parameters.prompt_suffix_token_ids must be a list of token IDs")
    required_sampling = {"temperature", "top_p", "top_k", "do_sample", "max_new_tokens"}
    missing_sampling = sorted(required_sampling - set(metadata["sampling_parameters"]))
    if missing_sampling:
        raise ValueError(f"fixed trajectory sampling_parameters is missing fields: {missing_sampling}")
    if int(metadata["sampling_parameters"]["max_new_tokens"]) != int(metadata["max_response_length"]):
        raise ValueError("sampling max_new_tokens must match fixed trajectory max_response_length")
    for key in (
        "tokenizer_fingerprint",
        "processor_template_fingerprint",
        "model_config_fingerprint",
    ):
        _require_sha256(metadata[key], key)
    for key in ("collector_policy_id", "reference_policy_id"):
        _require_portable_identifier(metadata[key], f"fixed trajectory metadata {key}")
    if expected:
        for key, value in expected.items():
            if key not in metadata:
                raise ValueError(f"cannot verify expected fixed trajectory metadata field {key!r}")
            if metadata[key] != value:
                raise ValueError(
                    f"fixed trajectory metadata mismatch for {key}: stored={metadata[key]!r}, expected={value!r}"
                )


def _validate_store(store: Mapping[str, Any], expected_metadata: Mapping[str, Any] | None = None) -> None:
    if store.get("format_version") != FORMAT_VERSION or store.get("contract") != CONTRACT_NAME:
        raise ValueError(
            "fixed COPO training requires a copo.fixed_trajectory.v3 store; "
            "legacy/v2 reward stores cannot be upgraded because they lack fixed-policy identity and log-probabilities"
        )
    metadata = store.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("fixed trajectory store metadata must be a mapping")
    _validate_metadata(metadata, expected_metadata)

    missing = [key for key in (*ROW_LIST_FIELDS, *ROW_TENSOR_FIELDS) if key not in store]
    if missing:
        raise ValueError(f"fixed trajectory store is missing required fields: {missing}")

    row_count = len(store["sample_ids"])
    if row_count == 0:
        raise ValueError("fixed trajectory store contains no trajectories")
    for key in ROW_LIST_FIELDS:
        if not isinstance(store[key], (list, tuple)) or len(store[key]) != row_count:
            raise ValueError(f"{key} must contain exactly {row_count} rows")
    for key in ROW_TENSOR_FIELDS:
        tensor = torch.as_tensor(store[key])
        if tensor.ndim == 0 or tensor.shape[0] != row_count:
            raise ValueError(f"{key} must have row dimension {row_count}, got {tuple(tensor.shape)}")

    response_ids = torch.as_tensor(store["response_token_ids"], dtype=torch.long)
    if response_ids.ndim != 2:
        raise ValueError("response_token_ids must be a 2D tensor")
    width = response_ids.shape[1]
    if width != int(metadata["max_response_length"]):
        raise ValueError("response token width does not match metadata max_response_length")
    for key in TOKEN_FIELDS:
        if torch.as_tensor(store[key]).shape != response_ids.shape:
            raise ValueError(f"{key} must have shape {tuple(response_ids.shape)}")

    prompt_ids = torch.as_tensor(store["prompt_token_ids"], dtype=torch.long)
    prompt_lengths = torch.as_tensor(store["prompt_lengths"], dtype=torch.long)
    if prompt_ids.ndim != 2 or prompt_lengths.shape != (row_count,):
        raise ValueError("prompt_token_ids/prompt_lengths have invalid shapes")

    response_lengths = torch.as_tensor(store["response_lengths"], dtype=torch.long)
    response_mask = torch.as_tensor(store["response_mask"], dtype=torch.bool)
    reasoning = torch.as_tensor(store["reasoning_mask"], dtype=torch.bool)
    answer = torch.as_tensor(store["answer_mask"], dtype=torch.bool)
    special = torch.as_tensor(store["special_token_mask"], dtype=torch.bool)
    for key in ("response_mask", "reasoning_mask", "answer_mask", "special_token_mask"):
        raw_mask = torch.as_tensor(store[key])
        if not torch.all((raw_mask == 0) | (raw_mask == 1)):
            raise ValueError(f"{key} must be binary")
    rollout_indices = torch.as_tensor(store["rollout_indices"], dtype=torch.long)
    terminal_ids = torch.as_tensor(store["terminal_token_ids"], dtype=torch.long)
    ends_with_eos = torch.as_tensor(store["ends_with_eos"], dtype=torch.bool)

    if response_lengths.shape != (row_count,) or rollout_indices.shape != (row_count,):
        raise ValueError("response_lengths/rollout_indices must be one-dimensional per-row tensors")
    if terminal_ids.shape != (row_count,) or ends_with_eos.shape != (row_count,):
        raise ValueError("terminal token metadata must be one-dimensional per-row tensors")

    old = torch.as_tensor(store["old_policy_log_probs"], dtype=torch.float32)
    ref = torch.as_tensor(store["reference_log_probs"], dtype=torch.float32)
    suff = torch.as_tensor(store["sufficiency"], dtype=torch.float32)
    nec = torch.as_tensor(store["necessity"], dtype=torch.float32)
    causal = torch.as_tensor(store["causal_reward"], dtype=torch.float32)
    outcomes = torch.as_tensor(store["outcome_rewards"], dtype=torch.float32)
    if outcomes.shape != (row_count,) or not torch.isfinite(outcomes).all():
        raise ValueError("outcome_rewards must contain one finite scalar per trajectory")

    eos_ids = {int(value) for value in metadata["eos_token_ids"]}
    pad_id = int(metadata["pad_token_id"])
    group_rows: dict[tuple[str, str], list[int]] = {}
    for row in range(row_count):
        for key in ("sample_ids", "group_ids", "image_ids"):
            _require_portable_identifier(store[key][row], key, row)
        prompt_len = int(prompt_lengths[row])
        response_len = int(response_lengths[row])
        if prompt_len < 1 or prompt_len > prompt_ids.shape[1]:
            raise ValueError(f"invalid prompt length {prompt_len} at row {row}")
        if torch.any(prompt_ids[row, prompt_len:] != pad_id):
            raise ValueError(f"prompt padding token mismatch at row {row}")
        if response_len < 1 or response_len > width:
            raise ValueError(f"invalid response length {response_len} at row {row}")
        expected_mask = torch.arange(width) < response_len
        if not torch.equal(response_mask[row].cpu(), expected_mask):
            raise ValueError(f"response_mask does not match response_length at row {row}")
        if (
            torch.any(reasoning[row] & answer[row])
            or torch.any(reasoning[row] & special[row])
            or torch.any(answer[row] & special[row])
        ):
            raise ValueError(f"reasoning/answer/special masks overlap at row {row}")
        for name, mask in (("reasoning_mask", reasoning), ("answer_mask", answer), ("special_token_mask", special)):
            if torch.any(mask[row] & ~response_mask[row]):
                raise ValueError(f"{name} selects padding at row {row}")
        if not reasoning[row].any() or not answer[row].any():
            raise ValueError(f"trajectory row {row} lacks a verified reasoning or answer span")
        if not special[row].any():
            raise ValueError(f"trajectory row {row} lacks explicit delimiter/EOS special tokens")
        if not torch.equal(reasoning[row] | answer[row] | special[row], response_mask[row]):
            raise ValueError(f"trajectory row {row} has unclassified response tokens")
        if torch.any(response_ids[row, response_len:] != pad_id):
            raise ValueError(f"response padding token mismatch at row {row}")
        if int(terminal_ids[row]) != int(response_ids[row, response_len - 1]):
            raise ValueError(f"terminal token does not match the last generated action at row {row}")
        if bool(ends_with_eos[row]) != (int(terminal_ids[row]) in eos_ids):
            raise ValueError(f"EOS metadata does not match terminal token at row {row}")
        if not str(store["stop_reasons"][row]).strip():
            raise ValueError(f"stop reason is empty at row {row}")
        if not torch.isfinite(old[row, response_mask[row]]).all() or not torch.isfinite(
            ref[row, response_mask[row]]
        ).all():
            raise ValueError(f"old/reference log-probabilities are non-finite at row {row}")
        if torch.all(old[row, response_mask[row]] == 0) or torch.all(ref[row, response_mask[row]] == 0):
            raise ValueError(f"old/reference log-probabilities cannot be all-zero placeholders at row {row}")
        if torch.any(old[row, ~response_mask[row]] != 0) or torch.any(ref[row, ~response_mask[row]] != 0):
            raise ValueError(f"old/reference log-probability padding must be zero at row {row}")
        allowed_causal = reasoning[row] & response_mask[row] & ~special[row]
        for name, tensor in (("sufficiency", suff), ("necessity", nec), ("causal_reward", causal)):
            if not torch.isfinite(tensor[row, response_mask[row]]).all():
                raise ValueError(f"{name} is non-finite at row {row}")
            if torch.any(tensor[row, ~allowed_causal] != 0):
                raise ValueError(f"{name} is nonzero outside verified reasoning tokens at row {row}")
        expected_causal = 0.35 * suff[row] + 0.35 * nec[row]
        if not torch.allclose(causal[row], expected_causal, atol=1e-6, rtol=1e-6):
            raise ValueError(f"causal_reward does not equal 0.35*sufficiency + 0.35*necessity at row {row}")

        actual_prompt_hash = token_sequence_hash(prompt_ids[row, :prompt_len].tolist())
        if store["prompt_hashes"][row] != actual_prompt_hash:
            raise ValueError(f"prompt hash does not match prompt tokens at row {row}")
        _require_sha256(store["image_hashes"][row], "image_hash", row)
        if str(store["collector_policy_ids"][row]) != str(metadata["collector_policy_id"]):
            raise ValueError(f"collector policy ID mismatch at row {row}")
        if str(store["reference_policy_ids"][row]) != str(metadata["reference_policy_id"]):
            raise ValueError(f"reference policy ID mismatch at row {row}")
        expected_id = make_trajectory_id(
            sample_id=store["sample_ids"][row],
            group_id=store["group_ids"][row],
            collection_round=int(metadata["collection_round"]),
            rollout_index=int(rollout_indices[row]),
            prompt_hash=store["prompt_hashes"][row],
            image_hash=store["image_hashes"][row],
            response_token_ids=response_ids[row, :response_len].tolist(),
            response_mask=response_mask[row, :response_len].long().tolist(),
            reasoning_mask=reasoning[row, :response_len].long().tolist(),
            answer_mask=answer[row, :response_len].long().tolist(),
            special_token_mask=special[row, :response_len].long().tolist(),
            collector_policy_id=metadata["collector_policy_id"],
            reference_policy_id=metadata["reference_policy_id"],
            old_policy_log_probs=old[row, :response_len].tolist(),
            reference_log_probs=ref[row, :response_len].tolist(),
            outcome_reward=float(outcomes[row]),
            sufficiency=suff[row, :response_len].tolist(),
            necessity=nec[row, :response_len].tolist(),
            causal_reward=causal[row, :response_len].tolist(),
        )
        if str(store["trajectory_ids"][row]) != expected_id:
            raise ValueError(f"trajectory identity checksum mismatch at row {row}")
        key = (str(store["sample_ids"][row]), str(store["group_ids"][row]))
        group_rows.setdefault(key, []).append(row)

    if len(set(str(value) for value in store["trajectory_ids"])) != row_count:
        raise ValueError("trajectory IDs must be globally unique")

    group_size = int(metadata["group_size"])
    expected_rollouts = list(range(group_size))
    for key, rows in group_rows.items():
        found = sorted(int(rollout_indices[row]) for row in rows)
        if len(rows) != group_size or found != expected_rollouts:
            raise ValueError(
                f"fixed trajectory group {key} must contain rollout indices {expected_rollouts}, got {found}"
            )
        prompt_values = {store["prompt_hashes"][row] for row in rows}
        image_values = {store["image_hashes"][row] for row in rows}
        if len(prompt_values) != 1 or len(image_values) != 1:
            raise ValueError(f"fixed trajectory group {key} mixes prompt or image identities")


@dataclass(frozen=True)
class FixedTrajectoryBatch:
    store: Mapping[str, Any]
    rows: tuple[int, ...]

    def _tensor(self, key: str) -> torch.Tensor:
        return torch.as_tensor(self.store[key])[list(self.rows)]

    def _list(self, key: str) -> list[Any]:
        values = self.store[key]
        return [values[row] for row in self.rows]

    def add_generation_inputs(self, data) -> None:
        if len(data) != len(self.rows):
            raise ValueError("fixed trajectory batch size does not match generation input batch")
        ids = np.empty(len(self.rows), dtype=object)
        outcomes = np.empty(len(self.rows), dtype=object)
        response_ids = self._tensor("response_token_ids")
        lengths = self._tensor("response_lengths").tolist()
        for index, length in enumerate(lengths):
            ids[index] = response_ids[index, : int(length)].tolist()
            outcomes[index] = float(self._tensor("outcome_rewards")[index])
        data.non_tensor_batch["copo_fixed_response_token_ids"] = ids
        data.non_tensor_batch["copo_fixed_outcome_reward"] = outcomes
        suffix = self.store["metadata"]["sampling_parameters"].get("prompt_suffix_token_ids", [])
        suffixes = np.empty(len(self.rows), dtype=object)
        for index in range(len(self.rows)):
            suffixes[index] = list(suffix)
        data.non_tensor_batch["copo_fixed_prompt_suffix_token_ids"] = suffixes

    def attach_to_dataproto(self, data) -> None:
        if len(data) != len(self.rows):
            raise ValueError("fixed trajectory batch size does not match rollout output batch")
        actual_responses = data.batch["responses"].detach().cpu().long()
        actual_mask = data.batch["response_mask"].detach().cpu().bool()
        stored_responses = self._tensor("response_token_ids").cpu().long()
        stored_mask = self._tensor("response_mask").cpu().bool()
        if actual_responses.shape != stored_responses.shape or not torch.equal(actual_responses, stored_responses):
            raise ValueError("fixed rollout response tokens changed during DataProto construction")
        if not torch.equal(actual_mask, stored_mask):
            raise ValueError("fixed rollout response mask changed during DataProto construction")
        actual_image_hashes = data.non_tensor_batch.get("copo_actual_image_hash")
        if actual_image_hashes is None:
            raise ValueError("fixed rollout output is missing a hash of the processed multimodal content")
        for index, (actual, expected) in enumerate(zip(actual_image_hashes, self._list("image_hashes"), strict=True)):
            if str(actual) != str(expected):
                raise ValueError(f"fixed trajectory processed image hash mismatch at batch row {index}")

        response_width = actual_responses.shape[1]
        prompt_attention = data.batch["attention_mask"][:, :-response_width].detach().cpu().bool()
        actual_prompts = data.batch["prompts"].detach().cpu().long()
        stored_prompts = self._tensor("prompt_token_ids").cpu().long()
        prompt_lengths = self._tensor("prompt_lengths").tolist()
        for index, length in enumerate(prompt_lengths):
            actual = actual_prompts[index][prompt_attention[index]].tolist()
            expected = stored_prompts[index, : int(length)].tolist()
            if actual != expected:
                raise ValueError(f"fixed trajectory prompt token mismatch at batch row {index}")

        device = data.batch["responses"].device
        tensor_mapping = {
            "old_log_probs": "old_policy_log_probs",
            "ref_log_prob": "reference_log_probs",
            "copo_sufficiency_rewards": "sufficiency",
            "copo_necessity_rewards": "necessity",
            "copo_causal_rewards": "causal_reward",
            "copo_reasoning_mask": "reasoning_mask",
            "copo_answer_mask": "answer_mask",
            "special_token_mask": "special_token_mask",
        }
        for target, source in tensor_mapping.items():
            data.batch[target] = self._tensor(source).detach().to(device)

        rm_scores = torch.zeros_like(data.batch["response_mask"], dtype=torch.float32)
        lengths = self._tensor("response_lengths").to(device).long()
        outcomes = self._tensor("outcome_rewards").to(device).float()
        rm_scores[torch.arange(len(self.rows), device=device), lengths - 1] = outcomes
        data.batch["rm_scores"] = rm_scores
        data.non_tensor_batch["copo_trajectory_id"] = np.asarray(self._list("trajectory_ids"), dtype=object)
        data.non_tensor_batch["copo_group_id"] = np.asarray(self._list("group_ids"), dtype=object)
        data.non_tensor_batch["copo_rollout_index"] = self._tensor("rollout_indices").cpu().numpy()
        data.non_tensor_batch.pop("copo_fixed_response_token_ids", None)
        data.non_tensor_batch.pop("copo_fixed_outcome_reward", None)
        data.non_tensor_batch.pop("copo_fixed_prompt_suffix_token_ids", None)
        data.meta_info["copo_fixed_trajectory"] = True


class FixedTrajectoryStore:
    def __init__(self, store: Mapping[str, Any], expected_metadata: Mapping[str, Any] | None = None):
        _validate_store(store, expected_metadata)
        self.store = store
        self.metadata = dict(store["metadata"])
        self._groups: dict[tuple[str, str], list[int]] = {}
        for row, (sample_id, group_id) in enumerate(zip(store["sample_ids"], store["group_ids"], strict=True)):
            self._groups.setdefault((str(sample_id), str(group_id)), []).append(row)

    @classmethod
    def load(cls, path: str | Path, expected_metadata: Mapping[str, Any] | None = None):
        try:
            store = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"failed to load fixed trajectory store {str(path)!r}: {exc}") from exc
        if not isinstance(store, Mapping):
            raise ValueError("fixed trajectory store must be a mapping")
        return cls(store, expected_metadata)

    def select_groups(
        self,
        sample_ids: Sequence[object],
        group_ids: Sequence[object],
        image_ids: Sequence[object],
        image_hashes: Sequence[object],
    ) -> FixedTrajectoryBatch:
        count = len(sample_ids)
        if any(len(values) != count for values in (group_ids, image_ids, image_hashes)):
            raise ValueError("sample/group/image identity arrays must have identical lengths")
        rows: list[int] = []
        rollout_indices = torch.as_tensor(self.store["rollout_indices"], dtype=torch.long)
        group_size = int(self.metadata["group_size"])
        for index in range(count):
            key = (str(sample_ids[index]), str(group_ids[index]))
            candidates = self._groups.get(key)
            if candidates is None:
                raise ValueError(f"fixed trajectory store has no group for sample_id={key[0]!r}, group_id={key[1]!r}")
            ordered = sorted(candidates, key=lambda row: int(rollout_indices[row]))
            if len(ordered) != group_size:
                raise ValueError(f"fixed trajectory group {key} has an invalid rollout count")
            for row in ordered:
                if str(self.store["image_ids"][row]) != str(image_ids[index]):
                    raise ValueError(f"fixed trajectory image ID mismatch for group {key}")
                if str(self.store["image_hashes"][row]) != str(image_hashes[index]):
                    raise ValueError(f"fixed trajectory image hash mismatch for group {key}")
            rows.extend(ordered)
        return FixedTrajectoryBatch(self.store, tuple(rows))
