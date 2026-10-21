from pathlib import Path

import pytest
import torch

from verl.implementations.minigpt4.config import MiniGPT4Config
from verl.implementations.minigpt4.mini_gpt4 import load_stage2_checkpoint


def _valid_config(tmp_path: Path) -> MiniGPT4Config:
    paths = {}
    for name in ("llama_model", "vit_pretrained_path", "q_former_model", "ckpt", "bert_model_path"):
        path = tmp_path / name
        path.mkdir() if name in {"llama_model", "bert_model_path"} else path.touch()
        paths[name] = str(path)
    return MiniGPT4Config(**paths)


def test_real_run_requires_every_weight_source(tmp_path):
    config = _valid_config(tmp_path)
    config.q_former_model = ""
    with pytest.raises(ValueError, match="q_former_model"):
        config.validate()


def test_real_run_rejects_missing_weight_path(tmp_path):
    config = _valid_config(tmp_path)
    config.ckpt = str(tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError, match="ckpt"):
        config.validate()


def test_valid_local_weight_layout_passes(tmp_path):
    _valid_config(tmp_path).validate()


def test_model_kwargs_propagate_requested_llama_dtype(tmp_path):
    config = _valid_config(tmp_path)
    config.dtype = "bfloat16"
    assert config.as_model_kwargs()["llama_dtype"] == "bfloat16"


def test_stage2_checkpoint_rejects_mismatched_language_model_projection(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    checkpoint = tmp_path / "stage2.pth"
    torch.save({"model": {"weight": torch.zeros(4, 3)}}, checkpoint)
    with pytest.raises(ValueError, match="same 7B/13B architecture"):
        load_stage2_checkpoint(model, checkpoint)


def test_stage2_checkpoint_requires_model_state_dict(tmp_path):
    checkpoint = tmp_path / "stage2.pth"
    torch.save({"weights": {}}, checkpoint)
    with pytest.raises(ValueError, match="must contain a 'model' state dict"):
        load_stage2_checkpoint(torch.nn.Linear(1, 1), checkpoint)
