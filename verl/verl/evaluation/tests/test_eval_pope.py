import pytest
import torch

from verl.evaluation.eval_pope import (
    VALID_SPLITS,
    _generated_continuation,
    _normalize_ground_truth,
    _select_transformers_model_class,
    compute_metrics,
    normalize_answer,
    validate_checkpoint,
    validate_lora_checkpoint,
)


def test_normalize_answer_prefers_answer_tag():
    assert normalize_answer("<answer>Yes</answer>") == "yes"
    assert normalize_answer("  <answer>  No  </answer>  ") == "no"


def test_normalize_answer_tag_with_reasoning_prefix():
    assert normalize_answer("Some reasoning <answer>yes</answer>") == "yes"


def test_normalize_answer_bare_token_fallback():
    assert normalize_answer("Yes") == "yes"
    assert normalize_answer("the answer is No.") == "no"


def test_normalize_answer_malformed_returns_none():
    assert normalize_answer("") is None
    assert normalize_answer("   ") is None
    assert normalize_answer("<answer>maybe</answer>") is None
    assert normalize_answer("the situation is unclear") is None
    assert normalize_answer("perhaps") is None


def test_normalize_answer_tag_partial_prefix():
    assert normalize_answer("<answer>yes indeed</answer>") == "yes"
    assert normalize_answer("<answer>no not at all</answer>") == "no"


def test_normalize_ground_truth():
    assert _normalize_ground_truth("yes") == "yes"
    assert _normalize_ground_truth("No") == "no"
    assert _normalize_ground_truth("YES, it is") == "yes"
    assert _normalize_ground_truth(None) is None
    assert _normalize_ground_truth("maybe") is None


def test_compute_metrics_perfect():
    metrics = compute_metrics(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2 and metrics["tn"] == 2


def test_compute_metrics_confusion_counts():
    # gt yes/no, pred yes/no
    metrics = compute_metrics(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"])
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_compute_metrics_zero_denominator_guards():
    # all predicted 'no' -> precision denominator zero, recall denominator zero
    metrics = compute_metrics(["yes", "yes"], ["no", "no"])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_compute_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(["yes"], ["yes", "no"])


def test_compute_metrics_empty():
    metrics = compute_metrics([], [])
    assert metrics["total"] == 0
    assert metrics["accuracy"] == 0.0


def test_validate_checkpoint_missing():
    with pytest.raises(SystemExit):
        validate_checkpoint("/nonexistent/copo/checkpoint")


def test_validate_checkpoint_missing_config(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    with pytest.raises(SystemExit):
        validate_checkpoint(str(checkpoint))


def test_validate_checkpoint_ok(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    assert validate_checkpoint(str(checkpoint)).endswith("ckpt")


def test_validate_lora_checkpoint_uses_peft_manifest(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    assert validate_lora_checkpoint(str(adapter)).endswith("adapter")


def test_validate_lora_checkpoint_rejects_full_model_manifest_only(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "config.json").write_text("{}")
    with pytest.raises(SystemExit):
        validate_lora_checkpoint(str(adapter))


def test_generated_continuation_excludes_decoder_only_prompt():
    output = torch.tensor([11, 12, 13, 21, 22])
    assert torch.equal(_generated_continuation(output, 3), torch.tensor([21, 22]))


def test_generated_continuation_keeps_encoder_decoder_output():
    output = torch.tensor([0, 21, 22])
    assert torch.equal(_generated_continuation(output, 7, is_encoder_decoder=True), output)


def test_transformers_model_class_selection_uses_image_text_mapping():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    from transformers.models.instructblip.configuration_instructblip import InstructBlipConfig
    from transformers.models.llava.configuration_llava import LlavaConfig

    assert _select_transformers_model_class(LlavaConfig()) is AutoModelForImageTextToText
    assert _select_transformers_model_class(InstructBlipConfig()) is AutoModelForImageTextToText
    assert _select_transformers_model_class(GPT2Config()) is AutoModelForCausalLM


def test_valid_splits_are_four():
    assert set(VALID_SPLITS) == {"default", "random", "popular", "adversarial"}
