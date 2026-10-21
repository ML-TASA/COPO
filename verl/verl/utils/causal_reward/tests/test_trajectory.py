import pytest
import torch

from verl.utils.causal_reward.tests.synthetic_v3 import clone_store, make_synthetic_v3_store, reorder_store
from verl.utils.causal_reward.trajectory import FixedTrajectoryStore, multimodal_content_hash


def test_valid_v3_store_loads_and_selects_by_identity(tmp_path):
    path = tmp_path / "synthetic-v3.pt"
    torch.save(make_synthetic_v3_store(), path)
    store = FixedTrajectoryStore.load(path, {"tokenizer_fingerprint": "a" * 64})
    selected = store.select_groups(["sample-0"], ["group-0"], ["image-0"], ["d" * 64])
    assert selected.rows == (0, 1)


def test_multimodal_content_hash_is_deterministic_and_content_sensitive():
    first = {"images": [torch.tensor([[1, 2], [3, 4]])], "videos": None}
    same = {"videos": None, "images": [torch.tensor([[1, 2], [3, 4]])]}
    changed = {"images": [torch.tensor([[1, 2], [3, 5]])], "videos": None}
    assert multimodal_content_hash(first) == multimodal_content_hash(same)
    assert multimodal_content_hash(first) != multimodal_content_hash(changed)


def test_row_reorder_does_not_change_rollout_order():
    store = FixedTrajectoryStore(reorder_store(make_synthetic_v3_store(), [1, 0]))
    selected = store.select_groups(["sample-0"], ["group-0"], ["image-0"], ["d" * 64])
    assert selected._tensor("rollout_indices").tolist() == [0, 1]
    assert selected._tensor("response_token_ids")[:, 0].tolist() == [21, 22]


def test_changed_response_token_is_rejected():
    value = clone_store(make_synthetic_v3_store())
    value["response_token_ids"][0, 0] = 999
    with pytest.raises(ValueError, match="trajectory identity checksum"):
        FixedTrajectoryStore(value)


def test_changed_outcome_reward_is_rejected_by_trajectory_identity():
    value = clone_store(make_synthetic_v3_store())
    value["outcome_rewards"][0] = -1
    with pytest.raises(ValueError, match="trajectory identity checksum"):
        FixedTrajectoryStore(value)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("group_ids", ["wrong", "group-0"], "trajectory identity|rollout indices"),
        ("rollout_indices", torch.tensor([1, 1]), "trajectory identity|rollout indices"),
        ("collector_policy_ids", ["wrong", "synthetic-collector-policy"], "collector policy ID"),
        ("reference_policy_ids", ["wrong", "synthetic-reference-policy"], "reference policy ID"),
    ],
)
def test_group_rollout_and_policy_identity_mismatches_are_rejected(field, replacement, message):
    value = clone_store(make_synthetic_v3_store())
    value[field] = replacement
    with pytest.raises(ValueError, match=message):
        FixedTrajectoryStore(value)


def test_tokenizer_fingerprint_mismatch_is_rejected():
    with pytest.raises(ValueError, match="metadata mismatch for tokenizer_fingerprint"):
        FixedTrajectoryStore(make_synthetic_v3_store(), {"tokenizer_fingerprint": "f" * 64})


@pytest.mark.parametrize("field", ["collector_policy_id", "reference_policy_id"])
def test_expected_policy_identity_mismatch_is_rejected(field):
    with pytest.raises(ValueError, match=f"metadata mismatch for {field}"):
        FixedTrajectoryStore(make_synthetic_v3_store(), {field: "different-policy"})


def test_sample_and_image_identity_mismatch_is_rejected_at_selection():
    store = FixedTrajectoryStore(make_synthetic_v3_store())
    with pytest.raises(ValueError, match="no group"):
        store.select_groups(["wrong"], ["group-0"], ["image-0"], ["d" * 64])
    with pytest.raises(ValueError, match="image hash mismatch"):
        store.select_groups(["sample-0"], ["group-0"], ["image-0"], ["e" * 64])


def test_legacy_v2_store_is_clearly_rejected():
    with pytest.raises(ValueError, match="legacy/v2"):
        FixedTrajectoryStore({"format_version": 2, "contract": "copo.precomputed.v2"})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["response_lengths"].__setitem__(0, 4), "response_mask"),
        (lambda value: value["response_mask"].__setitem__((0, 5), 1), "response_mask"),
        (lambda value: value["terminal_token_ids"].__setitem__(0, 9), "terminal token"),
        (lambda value: value["ends_with_eos"].__setitem__(0, False), "EOS metadata"),
        (lambda value: value["reasoning_mask"].__setitem__((0, 2), 1), "masks overlap"),
        (lambda value: value["special_token_mask"].__setitem__((0, 1), 0), "unclassified"),
    ],
)
def test_length_eos_and_mask_mismatches_are_rejected(mutate, message):
    value = clone_store(make_synthetic_v3_store())
    mutate(value)
    with pytest.raises(ValueError, match=message):
        FixedTrajectoryStore(value)


def test_zero_placeholder_log_probs_are_rejected():
    value = clone_store(make_synthetic_v3_store())
    value["old_policy_log_probs"][0] = 0
    with pytest.raises(ValueError, match="all-zero placeholders"):
        FixedTrajectoryStore(value)


def test_local_absolute_image_path_is_not_accepted_as_public_identity():
    value = clone_store(make_synthetic_v3_store())
    value["image_ids"] = ["/private/dataset/image.jpg", "/private/dataset/image.jpg"]
    with pytest.raises(ValueError, match="local absolute path"):
        FixedTrajectoryStore(value)


def test_missing_verified_answer_span_is_rejected_instead_of_treating_all_tokens_as_reasoning():
    value = clone_store(make_synthetic_v3_store())
    value["answer_mask"].zero_()
    with pytest.raises(ValueError, match="lacks a verified reasoning or answer span"):
        FixedTrajectoryStore(value)


def test_invalid_fixed_prompt_suffix_metadata_is_rejected():
    value = clone_store(make_synthetic_v3_store())
    value["metadata"]["sampling_parameters"]["prompt_suffix_token_ids"] = [10, "bad"]
    with pytest.raises(ValueError, match="prompt_suffix_token_ids"):
        FixedTrajectoryStore(value)
