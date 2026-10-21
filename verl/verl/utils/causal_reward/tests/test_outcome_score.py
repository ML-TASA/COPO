"""User-facing validation of the COPO outcome (yes/no) reward definition."""

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.outcome import compute_score


def test_outcome_yes_reward_match():
    assert compute_score("<answer>Yes.</answer>", "yes") == 1


def test_outcome_no_reward_match():
    assert compute_score("Nope", "no") == 1


def test_outcome_yes_no_mismatch_zero():
    assert compute_score("no", "yes") == 0


def test_outcome_exact_match():
    assert compute_score("red car", "red car") == 1


def test_outcome_exact_mismatch_zero():
    assert compute_score("blue car", "red car") == 0


def test_outcome_caption_accepts_grounded_coco_objects():
    assert compute_score("A dog sits beside a person.", ["dog", "person"], task="caption") == 1


def test_outcome_caption_rejects_hallucinated_coco_objects():
    assert compute_score("A dog sits beside a bicycle.", ["dog", "person"], task="caption") == 0


def test_outcome_caption_requires_object_annotations():
    assert compute_score("A dog sits outside.", "unused", task="caption") == 0


def test_outcome_caption_uses_aliases_and_metadata():
    extra = {"task": "caption", "target_objects": ["person", "bicycle"]}
    assert compute_score("Two men stand by their bikes.", "unused", extra_info=extra) == 1


def test_outcome_empty_output_zero():
    assert compute_score("", "yes") == 0


def test_outcome_empty_label_raises():
    with pytest.raises(ValueError, match="nonempty"):
        compute_score("yes", "")


def test_outcome_normalization_punctuation_case():
    assert compute_score("  RED, CAR! ", "red car") == 1


def test_data_source_outcome_dispatch():
    assert default_compute_score("outcome", "yes", "yes") == 1


def test_unknown_data_source_still_raises():
    with pytest.raises(NotImplementedError):
        default_compute_score("unknown_source", "yes", "yes")
