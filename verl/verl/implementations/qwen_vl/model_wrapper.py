"""Forward wrapper that makes the Qwen-VL-Chat actor consume ``pixel_values``."""

from __future__ import annotations

from types import MethodType
from typing import Any, Optional


def wrap_qwen_vl_for_verl(model: Any) -> Any:
    """Patch a loaded ``QWenLMHeadModel`` to accept verl-style ``pixel_values``.

    Returns the same model instance (mutated in place).  Safe to call only on
    the Qwen-VL-Chat actor/ref; the caller is responsible for detection.
    """
    transformer = getattr(model, "transformer", None)
    visual = getattr(transformer, "visual", None)
    if visual is None or not hasattr(visual, "encode"):
        raise TypeError(
            "wrap_qwen_vl_for_verl expected a QWenLMHeadModel with "
            "transformer.visual (Qwen-VL-Chat), but the model does not expose "
            f"that structure (got model={type(model).__name__})."
        )

    original_visual_encode = visual.encode
    pixel_values_box: dict[str, Optional[Any]] = {"tensor": None}

    def encode_from_pixels(paths):
        pixel_values = pixel_values_box.get("tensor")
        if pixel_values is not None:
            return visual(pixel_values)
        return original_visual_encode(paths)

    visual.encode = encode_from_pixels

    original_forward = model.forward
    original_prepare_inputs = getattr(model, "prepare_inputs_for_generation", None)

    def forward(input_ids=None, pixel_values=None, attention_mask=None, *args, **kwargs):
        if pixel_values is not None:
            pixel_values_box["tensor"] = pixel_values
        try:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                *args,
                **kwargs,
            )
        finally:
            pixel_values_box["tensor"] = None

    model.forward = forward

    if original_prepare_inputs is not None:
        def prepare_inputs_for_generation(
            _self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            **kwargs,
        ):
            model_inputs = original_prepare_inputs(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            if past_key_values is None and pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
            return model_inputs

        model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, model)

    # Qwen-VL-Chat predates transformers Cache objects and returns the legacy
    # tuple-of-tuples format.  Prevent modern GenerationMixin versions from
    # injecting an empty DynamicCache that the remote-code model cannot read.
    model._supports_default_dynamic_cache = MethodType(lambda _self: False, model)
    return model
