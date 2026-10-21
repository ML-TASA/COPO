"""Minimal ``(image, instruction, response)`` sample surface for MiniGPT-4."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class MiniGPT4Sample:
    sample_id: str
    image_path: str
    instruction: str
    response: str = ""
    answer: object = ""
    task: str = "vqa"
    group_id: str = ""
    image_id: str = ""


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
