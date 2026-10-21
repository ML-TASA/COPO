from __future__ import annotations

import copy

import torch

from verl.utils.causal_reward.trajectory import (
    CONTRACT_NAME,
    FORMAT_VERSION,
    POLICY_LOGPROB_SEMANTICS,
    make_trajectory_id,
    token_sequence_hash,
)


def make_synthetic_v3_store() -> dict:
    """Build a small contract-valid test fixture, not a collected reward store."""
    prompt_ids = torch.tensor([[11, 12, 0], [11, 12, 0]], dtype=torch.long)
    prompt_lengths = torch.tensor([2, 2], dtype=torch.long)
    response_ids = torch.tensor(
        [[21, 101, 31, 102, 2, 0], [22, 101, 32, 102, 2, 0]], dtype=torch.long
    )
    lengths = torch.tensor([5, 5], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long)
    reasoning_mask = torch.tensor([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.long)
    answer_mask = torch.tensor([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.long)
    special_mask = torch.tensor([[0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0]], dtype=torch.long)
    old = torch.tensor(
        [[-1.00, -1.10, -1.20, -1.30, -1.40, 0.0], [-1.05, -1.15, -1.25, -1.35, -1.45, 0.0]],
        dtype=torch.float32,
    )
    reference = torch.tensor(
        [[-1.20, -1.30, -1.40, -1.50, -1.60, 0.0], [-1.25, -1.35, -1.45, -1.55, -1.65, 0.0]],
        dtype=torch.float32,
    )
    sufficiency = torch.tensor([[0.4, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]], dtype=torch.float32)
    necessity = torch.tensor([[0.6, 0, 0, 0, 0, 0], [-0.1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    causal = 0.35 * sufficiency + 0.35 * necessity
    prompt_hash = token_sequence_hash([11, 12])
    image_hash = "d" * 64
    collector = "synthetic-collector-policy"
    reference_id = "synthetic-reference-policy"

    store = {
        "contract": CONTRACT_NAME,
        "format_version": FORMAT_VERSION,
        "metadata": {
            "model_family": "llava15",
            "tokenizer_fingerprint": "a" * 64,
            "processor_template_fingerprint": "b" * 64,
            "collector_policy_id": collector,
            "reference_policy_id": reference_id,
            "collection_round": 0,
            "group_size": 2,
            "model_config_fingerprint": "c" * 64,
            "sampling_parameters": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "do_sample": True,
                "max_new_tokens": 6,
            },
            "log_prob_semantics": POLICY_LOGPROB_SEMANTICS,
            "max_response_length": 6,
            "lambda_s": 0.35,
            "lambda_n": 0.35,
            "eos_token_ids": [2],
            "pad_token_id": 0,
        },
        "sample_ids": ["sample-0", "sample-0"],
        "group_ids": ["group-0", "group-0"],
        "prompt_hashes": [prompt_hash, prompt_hash],
        "image_ids": ["image-0", "image-0"],
        "image_hashes": [image_hash, image_hash],
        "stop_reasons": ["eos", "eos"],
        "collector_policy_ids": [collector, collector],
        "reference_policy_ids": [reference_id, reference_id],
        "rollout_indices": torch.tensor([0, 1]),
        "prompt_token_ids": prompt_ids,
        "prompt_lengths": prompt_lengths,
        "response_token_ids": response_ids,
        "response_lengths": lengths,
        "response_mask": response_mask,
        "reasoning_mask": reasoning_mask,
        "answer_mask": answer_mask,
        "special_token_mask": special_mask,
        "terminal_token_ids": torch.tensor([2, 2]),
        "ends_with_eos": torch.tensor([True, True]),
        "old_policy_log_probs": old,
        "reference_log_probs": reference,
        "outcome_rewards": torch.tensor([1.0, 0.0]),
        "sufficiency": sufficiency,
        "necessity": necessity,
        "causal_reward": causal,
    }
    store["trajectory_ids"] = [
        make_trajectory_id(
            sample_id=store["sample_ids"][row],
            group_id=store["group_ids"][row],
            collection_round=0,
            rollout_index=int(store["rollout_indices"][row]),
            prompt_hash=prompt_hash,
            image_hash=image_hash,
            response_token_ids=response_ids[row, : lengths[row]].tolist(),
            response_mask=response_mask[row, : lengths[row]].tolist(),
            reasoning_mask=reasoning_mask[row, : lengths[row]].tolist(),
            answer_mask=answer_mask[row, : lengths[row]].tolist(),
            special_token_mask=special_mask[row, : lengths[row]].tolist(),
            collector_policy_id=collector,
            reference_policy_id=reference_id,
            old_policy_log_probs=old[row, : lengths[row]].tolist(),
            reference_log_probs=reference[row, : lengths[row]].tolist(),
            outcome_reward=float(store["outcome_rewards"][row]),
            sufficiency=sufficiency[row, : lengths[row]].tolist(),
            necessity=necessity[row, : lengths[row]].tolist(),
            causal_reward=causal[row, : lengths[row]].tolist(),
        )
        for row in range(2)
    ]
    return store


def clone_store(store: dict) -> dict:
    return copy.deepcopy(store)


def reorder_store(store: dict, order: list[int]) -> dict:
    result = clone_store(store)
    row_fields = (
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
    tensor_fields = tuple(key for key, value in store.items() if isinstance(value, torch.Tensor))
    for key in row_fields:
        result[key] = [store[key][index] for index in order]
    for key in tensor_fields:
        result[key] = store[key][order]
    return result
