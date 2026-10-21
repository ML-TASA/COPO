"""Tests for the Qwen-VL input adapter + processor (no model download required)."""

from types import SimpleNamespace

import pytest

from verl.implementations.qwen_vl import (
    QWenVLProcessor,
    dedup_qwen_vl_image_tokens,
    is_qwen_vl,
    resolve_image_path,
)
from verl.utils.tokenizer import _create_qwen_vl_processor

IMG_START, IMG_PAD, IMG_END = 151857, 151859, 151858


class _FakeTokenizer:
    """Minimal QWenTokenizer stand-in: carries the img token ids and is callable."""

    def __init__(self, img_start_id=None, img_pad_id=None, img_end_id=None):
        self.img_start_id = img_start_id
        self.img_pad_id = img_pad_id
        self.img_end_id = img_end_id

    def __call__(self, text, **kwargs):
        # The real QWenTokenizer also emits an all-zero ``token_type_ids`` (a
        # BERT-style segment-id artifact); mirror that so the processor strip is
        # exercised by tests.
        return {
            "input_ids": [[1, 2, 3]] * len(text),
            "attention_mask": [[1, 1, 1]] * len(text),
            "token_type_ids": [[0, 0, 0]] * len(text),
        }


def _qwen_vl_tokenizer():
    return _FakeTokenizer(img_start_id=IMG_START, img_pad_id=IMG_PAD, img_end_id=IMG_END)


def _plain_tokenizer():
    return _FakeTokenizer()  # no img_* attributes -> not Qwen-VL


def test_is_qwen_vl_detects_legacy_checkpoint_only():
    assert is_qwen_vl(SimpleNamespace(model_type="qwen", architectures=["QWenLMHeadModel"]))
    # Qwen2.5-VL must never match.
    assert not is_qwen_vl(SimpleNamespace(model_type="qwen2_5_vl", architectures=["Qwen2_5_VLForConditionalGeneration"]))
    assert not is_qwen_vl(SimpleNamespace(model_type="qwen", architectures=["Qwen2ForCausalLM"]))


def test_dedup_collapses_padded_image_span():
    # Prompt: [BOS] <img> pad*256 </img> [text...]  (padded layout the actor uses)
    prompt = [1, IMG_START] + [IMG_PAD] * 256 + [IMG_END, 2, 3, 4]
    out = dedup_qwen_vl_image_tokens(prompt, _qwen_vl_tokenizer())
    assert out == [1, IMG_START, IMG_END, 2, 3, 4]
    assert IMG_PAD not in out


def test_dedup_handles_multiple_images():
    prompt = (
        [IMG_START] + [IMG_PAD] * 256 + [IMG_END, 10]
        + [IMG_START] + [IMG_PAD] * 256 + [IMG_END, 11]
    )
    out = dedup_qwen_vl_image_tokens(prompt, _qwen_vl_tokenizer())
    assert out == [IMG_START, IMG_END, 10, IMG_START, IMG_END, 11]


def test_dedup_is_noop_for_non_qwen_vl_tokenizer():
    prompt = [IMG_START, IMG_PAD, IMG_END, 5]
    assert dedup_qwen_vl_image_tokens(prompt, _plain_tokenizer()) == prompt


def test_resolve_image_path_relative_and_absolute(tmp_path):
    assert resolve_image_path("img.png", image_root=tmp_path) == str((tmp_path / "img.png").resolve())
    abs_path = tmp_path / "abs.png"
    assert resolve_image_path(str(abs_path), image_root=tmp_path) == str(abs_path)
    with pytest.raises(ValueError, match="image_root"):
        resolve_image_path("img.png")


def _config():
    return SimpleNamespace(visual={"image_size": 448, "patch_size": 14, "image_start_id": IMG_START})


def test_processor_apply_chat_template_picture_prefix():
    processor = QWenVLProcessor(tokenizer=_qwen_vl_tokenizer(), config=_config())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "ignored"},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    assert "Picture 1: <img></img>\n" in prompt
    assert "What is this?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    # Image marker must be unpadded text so the tokenizer pads it (and vLLM dedups it).
    assert "<img></img>" in prompt


def test_processor_call_returns_pixel_values():
    from PIL import Image

    processor = QWenVLProcessor(tokenizer=_qwen_vl_tokenizer(), config=_config())
    out = processor(text=["Picture 1: <img></img>\nHi"], images=[Image.new("RGB", (64, 64))], return_tensors="pt")
    assert out["pixel_values"].shape[1:] == (3, 448, 448)
    assert out["pixel_values"].shape[0] == 1


def test_processor_call_strips_token_type_ids():
    # The legacy QWenTokenizer emits an all-zero ``token_type_ids``.  verl's
    # agent loop forwards every non-input_ids/attention_mask key into the actor,
    # where QWenModel.forward would reshape it against the full prompt+response
    # length and crash.  The processor must drop it.
    processor = QWenVLProcessor(tokenizer=_qwen_vl_tokenizer(), config=_config())
    out = processor(text=["Picture 1: <img></img>\nHi"], return_tensors="pt")
    assert "token_type_ids" not in out


def test_processor_rejects_video():
    processor = QWenVLProcessor(tokenizer=_qwen_vl_tokenizer(), config=_config())
    with pytest.raises(ValueError, match="video"):
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video", "video": "x"}]}], add_generation_prompt=False
        )


def test_qwen_processor_construction_failure_is_not_silent(monkeypatch):
    import transformers

    config = _config()
    config.model_type = "qwen"
    config.architectures = ["QWenLMHeadModel"]
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    def fail_tokenizer(*args, **kwargs):
        raise OSError("missing tokenizer files")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail_tokenizer)
    with pytest.raises(RuntimeError, match="refusing to continue without multimodal inputs"):
        _create_qwen_vl_processor("broken-qwen-vl", {"trust_remote_code": True})
