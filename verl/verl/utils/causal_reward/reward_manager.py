import hashlib
import importlib
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from verl.utils.causal_reward.generation import GenerationResult, outcome_reward, validate_generation_result
from verl.utils.causal_reward.integration import build_causal_reward_tensor, causal_reward_metrics

logger = logging.getLogger(__name__)


REQUIRED_BATCH_FIELDS = ("responses", "response_mask")
REWARD_FIELDS = ("copo_causal_rewards", "copo_sufficiency_rewards", "copo_necessity_rewards")


def validate_batch_contract(data, require_reward=False):
    missing = [key for key in REQUIRED_BATCH_FIELDS if key not in data.batch]
    if missing:
        raise ValueError(f"COPO batch is missing required tensor fields: {missing}")
    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    if responses.shape != response_mask.shape:
        raise ValueError("responses and response_mask must have identical shapes")
    if require_reward and "copo_causal_rewards" not in data.batch:
        raise ValueError("copo.reward_source=precomputed requires copo_causal_rewards")
    for key in REWARD_FIELDS:
        if key in data.batch and data.batch[key].shape != response_mask.shape:
            raise ValueError(f"{key} must match response_mask shape {tuple(response_mask.shape)}")
    return True


def _find_subsequence(values, pattern, start=0):
    pattern = list(pattern)
    if not pattern:
        return None
    values = list(values)
    stop = len(values) - len(pattern) + 1
    for index in range(start, max(start, stop)):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def build_reasoning_answer_masks(responses, response_mask, answer_start_ids=(), answer_end_ids=()):
    reasoning = torch.zeros_like(response_mask)
    answer = torch.zeros_like(response_mask)
    fallbacks = 0
    starts = tuple(int(token) for token in answer_start_ids)
    ends = tuple(int(token) for token in answer_end_ids)
    for row in range(responses.shape[0]):
        valid_positions = torch.where(response_mask[row].bool())[0]
        valid_ids = [int(responses[row, pos]) for pos in valid_positions.tolist()]
        split_index = _find_subsequence(valid_ids, starts)
        end_index = None if split_index is None else _find_subsequence(
            valid_ids, ends, start=split_index + len(starts)
        )
        if split_index is None or end_index is None:
            fallbacks += 1
        else:
            reasoning[row, valid_positions[:split_index]] = 1
            answer_start = split_index + len(starts)
            answer_positions = valid_positions[answer_start:end_index]
            answer[row, answer_positions] = 1
    if fallbacks:
        logger.warning(
            "COPO response structure is invalid for %s samples; causal reward is disabled for those responses",
            fallbacks,
        )
    return reasoning, answer, fallbacks


def sufficiency_score(with_rewards, without_rewards):
    with_values = torch.as_tensor(with_rewards, dtype=torch.float32)
    without_values = torch.as_tensor(without_rewards, dtype=torch.float32)
    if with_values.shape != without_values.shape:
        raise ValueError(
            "sufficiency continuations with and without the token must have identical shapes"
        )
    if with_values.numel() == 0:
        raise ValueError("sufficiency requires at least one continuation pair")
    return torch.clamp(with_values - without_values, min=0).mean()


def necessity_score(original_reward, masked_reward):
    return torch.as_tensor(original_reward, dtype=torch.float32) - torch.as_tensor(masked_reward, dtype=torch.float32)


def mask_token(token_ids, index, valid_mask, mode="zero"):
    if token_ids.shape != valid_mask.shape:
        raise ValueError("token ids and valid mask must have identical shapes")
    masked = token_ids.clone()
    if valid_mask[index]:
        if mode == "zero":
            masked[index] = 0
        elif mode == "mean":
            masked[index] = token_ids[valid_mask].float().mean().round().to(token_ids.dtype)
        elif mode == "noise":
            masked[index] = torch.randint(0, max(int(token_ids.max()) + 1, 1), (), device=token_ids.device)
        else:
            raise ValueError(f"invalid masking mode: {mode}")
    return masked


def build_necessity_intervention(
    response,
    response_mask,
    reasoning_mask,
    token_index,
    answer_start_ids=(),
    mode="zero",
):
    """Build the valid, answer-free prefix used for a necessity intervention."""
    valid_positions = torch.where(response_mask.bool())[0]
    matches = torch.where(valid_positions == int(token_index))[0]
    if matches.numel() != 1 or not bool(reasoning_mask[int(token_index)]):
        raise ValueError(f"necessity token index {token_index} is not a valid reasoning token")
    local_index = int(matches.item())

    reasoning_count = int((reasoning_mask.bool() & response_mask.bool()).sum().item())
    delimiter_length = len(tuple(answer_start_ids))
    if delimiter_length and reasoning_count + delimiter_length <= len(valid_positions):
        prefix_length = reasoning_count + delimiter_length
    else:
        # Delimiter fallback: retain all real response tokens, but never store
        # padding.  The caller records the fallback so it cannot be mistaken for
        # a verified reasoning/answer split.
        prefix_length = len(valid_positions)
    if local_index >= prefix_length:
        raise ValueError(f"necessity token index {token_index} falls outside the reasoning prefix")

    prefix = response[valid_positions[:prefix_length]].clone()
    return mask_token(prefix, local_index, torch.ones_like(prefix, dtype=torch.bool), mode)


def reward_cache_key(sample_id, prompt, response, model_identifier, config):
    payload = {
        "sample_id": str(sample_id),
        "prompt": str(prompt),
        "response": torch.as_tensor(response).tolist(),
        "model": str(model_identifier),
        "config": config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class RewardCache:
    def __init__(self, enabled=False, path=None):
        self.enabled = enabled
        self.path = Path(path).expanduser() if path else None
        if self.enabled and self.path is None:
            raise ValueError("enabled COPO reward cache requires copo.reward_cache.path")
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, key):
        return self.path / f"{key}.json"

    def get(self, key, shape):
        if not self.enabled or not self._file(key).exists():
            return None
        try:
            value = json.loads(self._file(key).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt COPO reward cache entry {key}: {exc}") from exc
        result = {name: torch.tensor(tensor, dtype=torch.float32) for name, tensor in value.items()}
        if any(tensor.shape != shape for tensor in result.values()):
            raise ValueError(f"cached COPO reward shape does not match {tuple(shape)}")
        return result

    def put(self, key, value):
        if self.enabled:
            payload = {name: tensor.detach().cpu().tolist() for name, tensor in value.items()}
            self._file(key).write_text(json.dumps(payload))


def load_generation_callable(path):
    if not path:
        raise ValueError("online COPO rewards require copo.generation_callable")
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


class COPORewardGenerationManager:
    def __init__(self, config, generator=None):
        self.config = config
        self.generator = generator
        cache_config = getattr(config, "reward_cache", SimpleNamespace(enable=False, path=None))
        if isinstance(cache_config, dict):
            cache_config = SimpleNamespace(**cache_config)
        self.cache = RewardCache(cache_config.enable, cache_config.path)
        self.counters = {"bad_samples": 0, "generation_failures": 0, "delimiter_fallbacks": 0, "cache_hits": 0, "cache_misses": 0}

    def prepare(self, data, algorithm_name):
        if algorithm_name == "grpo":
            return data, {}
        if algorithm_name != "copo":
            raise ValueError(f"algorithm.name must be grpo or copo, got {algorithm_name}")
        validate_batch_contract(data)
        source = self.config.reward_source
        if source not in {"precomputed", "online", "auto"}:
            raise ValueError(f"invalid copo.reward_source: {source}")
        existing = "copo_causal_rewards" in data.batch
        if source == "precomputed":
            if not existing:
                from verl.utils.causal_reward.precomputed import inject_precomputed_reward

                info = inject_precomputed_reward(data, self.config)
            else:
                info = {}
            validate_batch_contract(data, require_reward=True)
            result_data, result = self._finalize(data, generated=False)
            result["copo/reward_source"] = "precomputed"
            result["copo/precomputed_reward_loaded"] = 1.0
            if info.get("copo_causal_rewards_loaded"):
                result["copo/causal_reward_loaded"] = 1.0
                shape = info.get("copo_causal_reward_shape")
                if shape:
                    result["copo/causal_reward_batch_size"] = float(shape[0])
                    result["copo/causal_reward_response_length"] = float(shape[1])
            return result_data, result
        if source == "auto" and existing:
            validate_batch_contract(data, require_reward=True)
            return self._finalize(data, generated=False)
        return self._generate(data)
    def _generate(self, data):
        data.batch.pop("copo_causal_rewards", None)
        if self.generator is None:
            self.generator = load_generation_callable(getattr(self.config, "generation_callable", None))
        start = time.perf_counter()
        responses = data.batch["responses"]
        response_mask = data.batch["response_mask"]
        reasoning, answer, fallbacks = build_reasoning_answer_masks(
            responses,
            response_mask,
            getattr(self.config, "answer_start_token_ids", ()),
            getattr(self.config, "answer_end_token_ids", ()),
        )
        self.counters["delimiter_fallbacks"] += fallbacks
        valid = response_mask.clone()
        for key in ("special_token_mask", "image_token_mask"):
            if key in data.batch:
                valid = valid * (1 - data.batch[key])
        sufficiency = torch.zeros_like(response_mask, dtype=torch.float32)
        necessity = torch.zeros_like(response_mask, dtype=torch.float32)
        reasoning_limit = getattr(self.config, "max_reasoning_tokens_for_copo", None)
        limit = getattr(self.config, "max_causal_tokens", None)
        limit = reasoning_limit if limit is None else min(limit, reasoning_limit) if reasoning_limit is not None else limit
        sample_ids = data.non_tensor_batch.get("sample_id", data.non_tensor_batch.get("uid"))
        if sample_ids is None:
            raise ValueError("online COPO reward generation requires sample_id or uid")
        prompts = data.non_tensor_batch.get("prompt", [""] * responses.shape[0])
        model_identifier = getattr(self.config, "generation_model", "")
        active_mask = reasoning * valid
        for row in range(responses.shape[0]):
            config_key = {key: value for key, value in vars(self.config).items() if key != "reward_cache"}
            key = reward_cache_key(sample_ids[row], prompts[row], responses[row], model_identifier, config_key)
            cached = self.cache.get(key, response_mask[row].shape)
            if cached is not None:
                self.counters["cache_hits"] += 1
                sufficiency[row] = cached["sufficiency"]
                necessity[row] = cached["necessity"]
                continue
            self.counters["cache_misses"] += 1
            positions = torch.where(active_mask[row].bool())[0].tolist()
            if limit is not None:
                positions = positions[:limit]
            try:
                original = self._score(self.generator(mode="original", row=row, data=data), row, data)
                for pos in positions:
                    if self.config.enable_sufficiency:
                        prefix_with = responses[row, : pos + 1]
                        prefix_without = responses[row, :pos]
                        with_rewards = self._score(self.generator(mode="sufficiency_with", row=row, token_index=pos, prefix=prefix_with, samples=self.config.num_suffix_samples, data=data), row, data)
                        without_rewards = self._score(self.generator(mode="sufficiency_without", row=row, token_index=pos, prefix=prefix_without, samples=self.config.num_suffix_samples, data=data), row, data)
                        sufficiency[row, pos] = sufficiency_score(with_rewards, without_rewards)
                    if self.config.enable_necessity:
                        masked = build_necessity_intervention(
                            responses[row],
                            response_mask[row],
                            reasoning[row],
                            pos,
                            getattr(self.config, "answer_start_token_ids", ()),
                            self.config.masking,
                        )
                        masked_reward = self._score(self.generator(mode="necessity", row=row, token_index=pos, masked_response=masked, masking=self.config.masking, data=data), row, data)
                        necessity[row, pos] = necessity_score(original, masked_reward)
            except Exception as exc:
                self.counters["generation_failures"] += 1
                self.counters["bad_samples"] += 1
                raise RuntimeError(f"COPO reward generation failed for sample {sample_ids[row]}: {exc}") from exc
            self.cache.put(key, {"sufficiency": sufficiency[row], "necessity": necessity[row]})
        data.batch["copo_sufficiency_rewards"] = sufficiency
        data.batch["copo_necessity_rewards"] = necessity
        data.batch["copo_reasoning_mask"] = reasoning
        data.batch["copo_answer_mask"] = answer
        data.batch["copo_valid_mask"] = valid
        metrics = {"copo/reward_generation_seconds": time.perf_counter() - start, **{f"copo/{key}": value for key, value in self.counters.items()}}
        return self._finalize(data, generated=True, metrics=metrics)

    def _score(self, value, row, data):
        if isinstance(value, (int, float, torch.Tensor)):
            return value
        if isinstance(value, list):
            return [self._score(item, row, data) for item in value]
        result = validate_generation_result(value)
        if not result.success:
            self.counters["generation_failures"] += 1
            maximum = getattr(self.config, "max_generation_failures", 0)
            if maximum >= 0 and self.counters["generation_failures"] > maximum:
                raise RuntimeError(f"COPO generation failures exceeded {maximum}: {result.error_reason}")
            return 0.0
        if not result.generated_text:
            self.counters.setdefault("empty_generations", 0)
            self.counters["empty_generations"] += 1
            return 0.0
        labels = data.non_tensor_batch.get("answer", data.non_tensor_batch.get("label"))
        if labels is None:
            reward_models = data.non_tensor_batch.get("reward_model")
            labels = [item.get("ground_truth") if isinstance(item, dict) else None for item in reward_models] if reward_models is not None else None
        if labels is None or labels[row] is None:
            raise ValueError("real COPO generation requires answer, label, or reward_model.ground_truth")
        tasks = data.non_tensor_batch.get("task")
        task = tasks[row] if tasks is not None else None
        return outcome_reward(result, labels[row], getattr(self.config, "answer_normalization", "strict"), task)

    def _finalize(self, data, generated, metrics=None):
        response_mask = data.batch["response_mask"]
        if "copo_reasoning_mask" not in data.batch:
            reasoning, answer, fallbacks = build_reasoning_answer_masks(
                data.batch["responses"],
                response_mask,
                getattr(self.config, "answer_start_token_ids", ()),
                getattr(self.config, "answer_end_token_ids", ()),
            )
            data.batch["copo_reasoning_mask"] = reasoning
            data.batch["copo_answer_mask"] = answer
            self.counters["delimiter_fallbacks"] += fallbacks
        valid = data.batch.get("copo_valid_mask", response_mask)
        data.batch["copo_valid_mask"] = valid
        causal = build_causal_reward_tensor(
            response_mask,
            self.config,
            causal_reward=data.batch.get("copo_causal_rewards"),
            sufficiency_reward=data.batch.get("copo_sufficiency_rewards"),
            necessity_reward=data.batch.get("copo_necessity_rewards"),
            reasoning_mask=data.batch["copo_reasoning_mask"],
            special_token_mask=data.batch.get("special_token_mask"),
            image_token_mask=data.batch.get("image_token_mask"),
        )
        data.batch["copo_causal_rewards"] = causal
        result = causal_reward_metrics(causal, response_mask)
        result["copo/reward_generated"] = float(generated)
        if metrics:
            result.update(metrics)
        return data, result
