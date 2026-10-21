# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""

import types
import warnings

__all__ = ["hf_tokenizer", "hf_processor", "resolve_vision_patch_size"]


def _read_field(value, name):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _class_name(value):
    return type(value).__name__ if value is not None else "None"


def _related_fields(value):
    if value is None:
        return []
    names = value.keys() if isinstance(value, dict) else dir(value)
    terms = ("patch", "vision", "image", "size", "token", "feature")
    return sorted(str(name) for name in names if any(term in str(name).lower() for term in terms))


def resolve_vision_patch_size(processor=None, model=None, config=None):
    candidates = (
        (processor, ("patch_size",)),
        (processor, ("image_processor", "patch_size")),
        (processor, ("config", "vision_config", "patch_size")),
        (processor, ("config", "image_config", "patch_size")),
        (model, ("config", "vision_config", "patch_size")),
        (model, ("config", "image_config", "patch_size")),
        (config, ("vision_config", "patch_size")),
        (config, ("image_config", "patch_size")),
    )
    for root, path in candidates:
        value = root
        for field in path:
            value = _read_field(value, field)
            if value is None:
                break
        if value is not None:
            try:
                patch_size = int(value)
            except (TypeError, ValueError):
                continue
            if patch_size > 0:
                return patch_size
    related = {
        "processor": _related_fields(processor),
        "processor.image_processor": _related_fields(_read_field(processor, "image_processor")),
        "processor.config": _related_fields(_read_field(processor, "config")),
        "model": _related_fields(model),
        "model.config": _related_fields(_read_field(model, "config")),
        "config": _related_fields(config),
    }
    raise ValueError(
        "Unable to resolve vision patch_size from processor/model config; check the processor or model config. "
        f"processor_class={_class_name(processor)}, model_class={_class_name(model)}, "
        f"config_class={_class_name(config)}, related_fields={related}"
    )


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def _apply_qwen_vl_tokenizer_patch_if_needed(tokenizer) -> None:
    """Prepare the legacy QWenTokenizer (Qwen-VL-Chat 2023) for verl.

    The 2023 Qwen-VL checkpoint ships no ``chat_template`` and no ``eos``/``pad``
    token ids; verl's agent loop calls ``tokenizer.apply_chat_template`` and
    ``tokenizer.pad`` before the processor is involved.  Gated on the Qwen-VL-only
    ``img_*`` token ids, so it is a no-op for Qwen2.5-VL and every other family.
    """
    if not (hasattr(tokenizer, "img_start_id") and hasattr(tokenizer, "img_end_id")):
        return
    from verl.implementations.qwen_vl import ensure_qwen_vl_tokenizer

    ensure_qwen_vl_tokenizer(tokenizer)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    # Qwen-VL-Chat's QWenTokenizer ships no eos/pad token ids, so attach them (and
    # a chat template) before set_pad_token_id() runs -- otherwise the latter warns
    # "pad_token_id is None. Now set to None" for the Qwen-VL case.
    _apply_qwen_vl_tokenizer_patch_if_needed(tokenizer)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoConfig, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor
        processor.config = config
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                processor.get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, processor)
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                processor.get_rope_index = types.MethodType(Qwen2_5_VLModel.get_rope_index, processor)
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                processor.get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, processor)
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                processor.get_rope_index = types.MethodType(Glm4vModel.get_rope_index, processor)
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case "LlavaProcessor":
                pass
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    # Qwen-VL-Chat (2023) ships no HF processor config; synthesize one so the
    # stock verl agent loop / dataset / actor path still receives a processor.
    if processor is None:
        processor = _create_qwen_vl_processor(name_or_path, kwargs)
    return processor


def _create_qwen_vl_processor(name_or_path, kwargs):
    """Return a synthetic QWenVLProcessor for the legacy Qwen-VL-Chat checkpoint, else None."""
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(name_or_path, **kwargs)
    except Exception:
        return None

    from verl.implementations.qwen_vl import QWenVLProcessor, is_qwen_vl

    if not is_qwen_vl(config):
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        return QWenVLProcessor(tokenizer=tokenizer, config=config)
    except Exception as exc:
        raise RuntimeError(
            "failed to construct the required legacy Qwen-VL processor for "
            f"{name_or_path!r}; refusing to continue without multimodal inputs"
        ) from exc
