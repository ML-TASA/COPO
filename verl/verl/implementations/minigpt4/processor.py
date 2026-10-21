"""Image preprocessing + prompt building for the MiniGPT-4 backend.

The image transform reproduces the official ``BlipImageEvalProcessor``
(``blip2_image_eval``, ``image_size=224``): bicubic resize to a square, then
normalize with the BLIP-2 ImageNet statistics.  The prompt builder reproduces
the official template (``###Human: <Img><ImageHere></Img> … ###Assistant:``)
which ``MiniGPT4.prompt_wrap`` splits on the ``<ImageHere>`` placeholder.
"""

from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from .config import MiniGPT4Config


class MiniGPT4Processor:
    def __init__(self, config: MiniGPT4Config):
        self.config = config
        # Official BlipImageEvalProcessor for image_size=224.
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (config.image_size, config.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def preprocess_image(self, image):
        """PIL image -> ``(3, H, W)`` normalized tensor."""
        return self.transform(image)

    def build_prompt(self, instruction: str) -> str:
        """Instruction -> full prompt with the ``<ImageHere>`` placeholder."""
        return self.config.prompt_template.format(
            f"<Img><ImageHere></Img> {instruction}"
        )

    def build_sample(self, image, instruction: str) -> dict:
        """Return the ``samples`` dict ``MiniGPT4.generate`` expects."""
        return {
            "image": self.preprocess_image(image).unsqueeze(0),
            "prompt": self.build_prompt(instruction),
        }
