from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image
from verl.utils.causal_reward.generator import load_collection_samples
from verl.utils.dataset.rl_dataset import RLHFDataset


SCRIPT = Path(__file__).resolve().parents[5] / "scripts" / "prepare_data" / "build_training_data.py"
SPEC = importlib.util.spec_from_file_location("copo_build_training_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(tmp_path, **overrides):
    image = tmp_path / "image.png"
    Image.new("RGB", (3, 2), (12, 34, 56)).save(image)
    row = {
        "sample_id": "sample-1", "group_id": "group-1", "image_id": "image-1",
        "image": str(image), "instruction": "Is there a cat?", "answer": "no", "task": "pope",
    }
    row.update(overrides)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n")
    return source


def test_builder_emits_shared_instruction_and_native_identities(tmp_path):
    source = _source(tmp_path)
    canonical = tmp_path / "canonical.jsonl"
    native = tmp_path / "native.json"
    assert MODULE.build_datasets(str(source), str(canonical), str(native)) == (1, 1)
    canonical_row = json.loads(canonical.read_text())
    native_row = json.loads(native.read_text())
    assert canonical_row["instruction"] in native_row["prompt"][0]["content"]
    assert "Example response:" in canonical_row["instruction"]
    assert canonical_row["answer"] == native_row["reward_model"]["ground_truth"]
    for key in ("sample_id", "group_id", "image_id", "image_hash"):
        assert canonical_row[key] == native_row[key]
    assert native_row["images"][0]["image"].startswith("file://")
    collected = load_collection_samples(str(canonical))
    assert collected[0]["instruction"] == canonical_row["instruction"]


def test_native_json_is_consumed_by_installed_rlhf_dataset(tmp_path):
    source = _source(tmp_path)
    canonical = tmp_path / "canonical.jsonl"
    native = tmp_path / "native.json"
    MODULE.build_datasets(str(source), str(canonical), str(native))
    dataset = RLHFDataset(
        str(native),
        tokenizer=object(),
        processor=object(),
        config={
            "filter_overlong_prompts": False,
            "cache_dir": str(tmp_path / "cache"),
            "image_key": "images",
            "prompt_key": "prompt",
        },
    )
    item = dataset[0]
    assert item["sample_id"] == "sample-1"
    assert item["group_id"] == "group-1"
    assert item["image_id"] == "image-1"
    assert item["reward_model"]["ground_truth"] == "no"
    content = item["raw_prompt"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "text"
    assert "<think>" in content[1]["text"]


def test_builder_never_fabricates_missing_answer(tmp_path):
    source = _source(tmp_path, answer="")
    with pytest.raises(ValueError, match="answer"):
        MODULE.build_datasets(str(source), str(tmp_path / "a"), str(tmp_path / "b"))


def test_builder_rejects_partial_delimiter_instruction(tmp_path):
    source = _source(tmp_path, instruction="Use <think> but no closing marker")
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.build_datasets(str(source), str(tmp_path / "a"), str(tmp_path / "b"))
