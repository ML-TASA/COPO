from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from verl.model_merger.base_model_merger import BaseModelMerger
from verl.utils import fsdp_utils


class _Merger(BaseModelMerger):
    def merge_and_save(self):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError


def _merger(tmp_path, alpha):
    merger = _Merger.__new__(_Merger)
    merger.config = SimpleNamespace(target_dir=str(tmp_path), lora_alpha=alpha)
    return merger


def _lora_state():
    return {
        "base_model.model.layers.0.q_proj.lora_A.default.weight": torch.ones(2, 4),
        "base_model.model.layers.0.q_proj.lora_B.default.weight": torch.ones(4, 2),
        "base_model.model.layers.0.q_proj.base_layer.weight": torch.ones(4, 4),
    }


def test_native_export_requires_real_lora_alpha(tmp_path):
    with pytest.raises(ValueError, match="positive training-time --lora-alpha"):
        _merger(tmp_path, None).save_lora_adapter(_lora_state())


def test_native_export_preserves_lora_alpha_and_adapter_weights(tmp_path):
    state = _lora_state()
    adapter_path = _merger(tmp_path, 16).save_lora_adapter(state)
    config = json.loads((tmp_path / "lora_adapter" / "adapter_config.json").read_text())
    assert adapter_path == str(tmp_path / "lora_adapter")
    assert config["lora_alpha"] == 16
    assert (tmp_path / "lora_adapter" / "adapter_model.safetensors").is_file()
    assert set(state) == {"layers.0.q_proj.weight"}


def test_fsdp_adapter_export_falls_back_when_layered_discovery_is_empty(monkeypatch):
    full_state = {"base_model.model.q_proj.lora_A.default.weight": torch.ones(2, 4)}
    expected = {"base_model.model.q_proj.lora_A.weight": torch.ones(2, 4)}

    monkeypatch.setattr(fsdp_utils, "layered_summon_lora_params", lambda _model: {})
    monkeypatch.setattr(fsdp_utils, "get_fsdp_full_state_dict", lambda *_args, **_kwargs: full_state)
    monkeypatch.setattr(fsdp_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        "peft.utils.save_and_load.get_peft_model_state_dict",
        lambda _model, state_dict: expected if state_dict is full_state else {},
    )

    actual = fsdp_utils.checkpoint_lora_params(object())
    assert actual.keys() == expected.keys()
    torch.testing.assert_close(actual[next(iter(actual))], next(iter(expected.values())))


def test_layered_adapter_export_discovers_lora_fsdp_leaves(monkeypatch):
    class _Leaf:
        def state_dict(self):
            return {"_fsdp_wrapped_module.weight": torch.ones(2, 4)}

    leaf = _Leaf()

    class _Root:
        def named_modules(self):
            yield "", self
            yield "_fsdp_wrapped_module.base_model.model.q_proj.lora_A.default", leaf

    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda module: 1 if module is leaf else 0)
    monkeypatch.setattr(fsdp_utils.FSDP, "summon_full_params", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(fsdp_utils, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setattr(
        "peft.utils.save_and_load.get_peft_model_state_dict",
        lambda _model, state_dict: {
            name.replace(".default", ""): value for name, value in state_dict.items() if "lora_" in name
        },
    )

    actual = fsdp_utils.layered_summon_lora_params(_Root())
    key = "base_model.model.q_proj.lora_A.weight"
    assert list(actual) == [key]
    assert tuple(actual[key].shape) == (2, 4)


def test_lora_target_module_serialization_preserves_regex():
    pattern = r".*language_model.*\.(q_proj|v_proj)"
    assert fsdp_utils.serializable_lora_target_modules(pattern) == pattern
    assert fsdp_utils.serializable_lora_target_modules({"q_proj", "v_proj"}) == ["q_proj", "v_proj"]
