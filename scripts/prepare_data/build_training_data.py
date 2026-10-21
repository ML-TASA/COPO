#!/usr/bin/env python3
"""Validate COPO samples and emit collector/adapter plus native-verl datasets."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "verl"))

from verl.utils.causal_reward.trajectory import multimodal_content_hash  # noqa: E402


FORMAT_INSTRUCTION = (
    " Follow this exact response protocol. Example response: "
    "<think>I inspect the relevant visual evidence.</think><answer>the final answer</answer>. "
    "Now provide your own image-grounded reasoning and final answer in the same format; "
    "do not omit or rename any delimiter."
)


def _load(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        for field in ("sample_id", "group_id", "image_id", "image", "instruction", "answer", "task"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"{path}:{line_number} is missing nonempty field {field!r}")
        image_path = pathlib.Path(row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"{path}:{line_number} image does not exist: {image_path}")
        if "<think>" not in row["instruction"] and "<answer>" not in row["instruction"]:
            row["instruction"] = row["instruction"].rstrip() + FORMAT_INSTRUCTION
        elif not all(marker in row["instruction"] for marker in ("<think>", "</think>", "<answer>", "</answer>")):
            raise ValueError(f"{path}:{line_number} contains an incomplete response-format instruction")
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            row["image_hash"] = multimodal_content_hash({"images": [image], "videos": None})
        row["image"] = str(image_path.resolve())
        rows.append(row)
    if not rows:
        raise ValueError("input dataset contains no samples")
    identities = [(str(row["sample_id"]), str(row["group_id"])) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("sample_id/group_id pairs must be unique in the source dataset")
    return rows


def build_datasets(input_path: str, canonical_output: str, native_output: str) -> tuple[int, int]:
    rows = _load(pathlib.Path(input_path))
    canonical_path = pathlib.Path(canonical_output)
    native_path = pathlib.Path(native_output)
    if canonical_path.suffix not in {".json", ".jsonl"}:
        raise ValueError("canonical output must use .json or .jsonl")
    if native_path.suffix != ".json":
        raise ValueError("native verl output must use .json (the installed RLHFDataset format)")
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    native_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    native_rows = []
    for index, row in enumerate(rows):
        native_rows.append(
            {
                "data_source": row["task"],
                "prompt": [{"role": "user", "content": "<image>" + row["instruction"]}],
                "images": [{"image": pathlib.Path(row["image"]).as_uri()}],
                "reward_model": {"style": "rule", "ground_truth": row["answer"]},
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "image_id": row["image_id"],
                "image_hash": row["image_hash"],
                "answer": row["answer"],
                "task": row["task"],
                "extra_info": {"index": index},
            }
        )
    native_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in native_rows))
    return len(rows), len(native_rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build validated COPO training datasets.")
    parser.add_argument("--input", required=True, help="Source JSONL with real task labels")
    parser.add_argument("--canonical-output", required=True, help="Collector/adapter JSONL output")
    parser.add_argument("--native-output", required=True, help="Native verl JSONL output")
    args = parser.parse_args(argv)
    build_datasets(args.input, args.canonical_output, args.native_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
