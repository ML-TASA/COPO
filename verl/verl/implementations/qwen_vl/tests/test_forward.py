"""Tests for the Qwen-VL forward wrapper (``model_wrapper.py``).

Uses a tiny fake model that mirrors QWenLMHeadModel's structure
(``transformer.visual`` with ``encode`` + ``forward``) so the wrapper can be
exercised without loading the 9.7B checkpoint.
"""

import inspect

import pytest
import torch
import torch.nn as nn

from verl.implementations.qwen_vl import wrap_qwen_vl_for_verl


class _FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1)  # give it a real dtype/device anchor
        self.encoded_paths = []

    def forward(self, x):
        # VisionTransformer.forward: [N, 3, 448, 448] -> [N, 256, 4096]
        return torch.zeros(x.shape[0], 256, 4096, dtype=x.dtype, device=x.device)

    def encode(self, paths):
        self.encoded_paths.append(list(paths))
        return torch.zeros(len(paths), 256, 4096)


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.visual = _FakeVisual()
        self.last_attention_mask = None

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Mirror QWenModel.forward's path-decode branch: it always calls
        # self.visual.encode(...) with the decoded paths when image tokens exist.
        self.last_attention_mask = attention_mask
        paths = [""] * input_ids.shape[0]
        images = self.transformer.visual.encode(paths)
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 10)
        return SimpleNamespace(logits=logits, images_shape=tuple(images.shape))

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": kwargs.get("attention_mask"),
        }


class SimpleNamespace:  # minimal namespace stand-in
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_wrapper_injects_pixel_values_and_preserves_logits():
    model = _FakeModel()
    wrap_qwen_vl_for_verl(model)

    pv = torch.randn(2, 3, 448, 448)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    out = model(input_ids=torch.zeros(2, 5, dtype=torch.long), pixel_values=pv, attention_mask=attention_mask)

    # The patched encode must consume pixel_values via visual.forward (no disk
    # path decode), producing [N, 256, 4096].
    assert model.transformer.visual.encoded_paths == []
    assert torch.equal(model.last_attention_mask, attention_mask)
    assert "attention_mask" in inspect.signature(model.forward).parameters
    assert out.images_shape == (2, 256, 4096)
    # output.logits keeps verl's exact [bsz, seq_len, vocab] contract.
    assert out.logits.shape == (2, 5, 10)


def test_wrapper_clears_pixel_values_after_forward():
    model = _FakeModel()
    wrap_qwen_vl_for_verl(model)

    pv = torch.randn(1, 3, 448, 448)
    model(input_ids=torch.zeros(1, 3, dtype=torch.long), pixel_values=pv)
    assert model.transformer.visual.encoded_paths == []

    # A subsequent text-only call (no pixel_values) must fall back to the
    # original visual.encode(paths), proving no pixel_values state leaked.
    model(input_ids=torch.zeros(1, 3, dtype=torch.long))
    assert model.transformer.visual.encoded_paths == [[""]]


def test_wrapper_preserves_pixels_for_prefill_and_uses_legacy_cache():
    model = _FakeModel()
    wrap_qwen_vl_for_verl(model)

    input_ids = torch.zeros(1, 3, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pixels = torch.randn(1, 3, 448, 448)
    prefill = model.prepare_inputs_for_generation(
        input_ids, attention_mask=attention_mask, pixel_values=pixels
    )
    cached = model.prepare_inputs_for_generation(
        input_ids, past_key_values=((torch.zeros(1), torch.zeros(1)),),
        attention_mask=attention_mask, pixel_values=pixels,
    )

    assert torch.equal(prefill["pixel_values"], pixels)
    assert "pixel_values" not in cached
    assert model._supports_default_dynamic_cache() is False


def test_wrapper_rejects_non_qwen_vl_structure():
    class Bad(nn.Module):
        def forward(self, input_ids=None, **kwargs):
            return SimpleNamespace(logits=torch.zeros(1, 1, 2))

    with pytest.raises(TypeError, match="transformer.visual"):
        wrap_qwen_vl_for_verl(Bad())
