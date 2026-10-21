"""Minimal dataset: ``(image, instruction, response)`` triplets."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class InstructBlipSample:
    sample_id: str
    image_path: str
    instruction: str
    response: str = ""
    answer: object = ""
    task: str = "vqa"
    group_id: str = ""
    image_id: str = ""


class InstructBlipDataset(Dataset):
    def __init__(self, samples: list[InstructBlipSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> InstructBlipSample:
        return self.samples[index]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_collate_fn(processor):
    def collate_fn(samples: list[InstructBlipSample]) -> dict:
        processed = [processor.process(load_image(s.image_path), s.instruction, s.response) for s in samples]
        return processor.collate(processed)

    return collate_fn
