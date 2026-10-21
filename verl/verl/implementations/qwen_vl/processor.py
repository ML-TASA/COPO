"""Processor for Qwen-VL-Chat (2023)."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BatchFeature, PreTrainedTokenizer

# Text markers understood by the QWenTokenizer.
IMG_START = "<img>"
IMG_END = "</img>"
PICTURE_PREFIX = "Picture {}: " + IMG_START + IMG_END + "\n"

# Image normalization constants from the Qwen-VL-Chat visual tower
# (visual.py / vLLM's QwenVLProcessor).
_IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMG_STD = (0.26862954, 0.26130258, 0.27577711)


def _resolve_image_size(config: Any) -> int:
    visual = getattr(config, "visual", None)
    if visual is None:
        raise ValueError("Qwen-VL config is missing the 'visual' section; cannot resolve image_size.")
    if isinstance(visual, dict):
        return int(visual["image_size"])
    return int(visual.image_size)


def _resolve_patch_size(config: Any) -> int:
    visual = getattr(config, "visual", None)
    if visual is None:
        raise ValueError("Qwen-VL config is missing the 'visual' section; cannot resolve patch_size.")
    if isinstance(visual, dict):
        return int(visual["patch_size"])
    return int(visual.patch_size)


class QWenVLProcessor:
    """Processor implementing the ``transformers`` surface verl requires for Qwen-VL-Chat."""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Any):
        self.tokenizer = tokenizer
        self.config = config

        image_size = _resolve_image_size(config)
        self.patch_size = _resolve_patch_size(config)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
            ]
        )

        self.chat_template = getattr(self, "chat_template", "")
        self.image_processor = None

    # ------------------------------------------------------------------ #
    # chat template                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iter_content_segments(message: dict) -> Iterable[tuple[str, Any]]:
        """Yield ``(kind, value)`` segments for a message's content.

        ``kind`` is one of ``"text"`` / ``"image"`` / ``"video"``.  Content may
        be a plain string (a single text segment) or a list of
        ``{"type": ..., ...}`` dicts as produced by verl's ``_build_messages``.
        """
        content = message.get("content", "")
        if isinstance(content, str):
            yield ("text", content)
            return
        for segment in content:
            if not isinstance(segment, dict):
                yield ("text", str(segment))
                continue
            yield (segment.get("type", "text"), segment)

    @staticmethod
    def _count_images(messages: List[dict]) -> int:
        total = 0
        for message in messages:
            for kind, _ in QWenVLProcessor._iter_content_segments(message):
                if kind == "image":
                    total += 1
        return total

    def _render_message(self, message: dict, image_offset: int) -> tuple[str, int]:
        """Render one message into Qwen-VL-Chat text; returns (text, next_image_offset)."""
        role = message.get("role", "user")
        parts: List[str] = []

        for kind, value in self._iter_content_segments(message):
            if kind == "image":
                parts.append(PICTURE_PREFIX.format(image_offset + 1))
                image_offset += 1
            elif kind == "video":
                raise ValueError("Qwen-VL-Chat does not support video inputs.")
            else:
                text = value if isinstance(value, str) else value.get("text", "")
                parts.append(text)

        body = "".join(parts)
        return f"<|im_start|>{role}\n{body}<|im_end|>\n", image_offset

    def apply_chat_template(
        self,
        messages: List[dict],
        tools: Optional[list] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> str:
        """Render ``messages`` into the Qwen-VL-Chat chat format.

        Images are emitted as ``Picture N: <img></img>\\n`` prefixes.  ``tokenize=True``
        is accepted for API compatibility but returns the rendered string.
        """
        if tools:
            raise ValueError("Qwen-VL-Chat has no tool-calling support; tools must be None.")

        rendered: List[str] = []
        image_offset = 0
        for message in messages:
            text, image_offset = self._render_message(message, image_offset)
            rendered.append(text)

        if add_generation_prompt:
            rendered.append("<|im_start|>assistant\n")

        prompt = "".join(rendered)
        if tokenize:
            return prompt
        return prompt

    # ------------------------------------------------------------------ #
    # tokenize + image features                                          #
    # ------------------------------------------------------------------ #
    def _normalize_images(self, images: Optional[List[Image.Image]]) -> torch.Tensor:
        if not images:
            return torch.empty((0, 3, self._image_size, self._image_size))

        pixel_values = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError(f"Qwen-VL processor expects PIL images, got {type(image)}.")
            pixel_values.append(self.image_transform(image.convert("RGB")))
        return torch.stack(pixel_values, dim=0)

    @property
    def _image_size(self) -> int:
        return _resolve_image_size(self.config)

    def __call__(
        self,
        text: Optional[Any] = None,
        images: Optional[List[Image.Image]] = None,
        videos: Optional[list] = None,
        video_metadata: Optional[list] = None,
        return_tensors: Optional[str] = None,
        do_sample_frames: Optional[bool] = None,
        **kwargs: Any,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if videos:
            raise ValueError("Qwen-VL-Chat does not support video inputs.")

        text_inputs = self.tokenizer(text)

        # Drop the unused token_type_ids emitted by the QWenTokenizer.
        text_inputs.pop("token_type_ids", None)

        if images:
            image_inputs = {"pixel_values": self._normalize_images(images)}
        else:
            image_inputs = {}

        return BatchFeature({**text_inputs, **image_inputs}, tensor_type=return_tensors)


# ------------------------------------------------------------------ #
# tokenizer chat template                                              #
# ------------------------------------------------------------------ #
# Chat template for the QWenTokenizer, which ships without one.
QWEN_VL_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def ensure_qwen_vl_tokenizer(tokenizer) -> bool:
    """Set a chat template and pad/eos tokens on the Qwen-VL tokenizer if missing.

    Returns ``True`` if anything was patched, ``False`` otherwise.
    """
    if not (hasattr(tokenizer, "img_start_id") and hasattr(tokenizer, "img_end_id")):
        return False

    patched = False
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = QWEN_VL_CHAT_TEMPLATE
        patched = True
    if getattr(tokenizer, "eos_token_id", None) is None:
        eos_id = getattr(tokenizer, "im_end_id", None)
        if eos_id is None:
            eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.eos_token_id = eos_id
        patched = True
    if getattr(tokenizer, "pad_token_id", None) is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = pad_id
        patched = True
    return patched
