"""Shared exact-action mechanics for decoder-only multimodal HF backends."""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from verl.utils.causal_reward.collector import CollectorIdentity, GeneratedAction, model_state_fingerprint
from verl.utils.causal_reward.trajectory import (
    fingerprint_model_config,
    fingerprint_processor_template,
    fingerprint_tokenizer,
    multimodal_content_hash,
)


def require_local_checkpoint(checkpoint: str, label: str = "checkpoint") -> str:
    """Fail before model initialization instead of triggering an implicit download."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist locally: {checkpoint}")
    return str(path)


def load_peft_adapter(model, adapter_path: str | None):
    if not adapter_path:
        return model
    adapter_path = require_local_checkpoint(adapter_path, "PEFT collector state")
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"PEFT collector state lacks adapter_config.json: {adapter_path}")
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)


def generation_eos_token_ids(model, tokenizer) -> tuple[int, ...]:
    """Return the stop IDs actually used by ``model.generate``.

    Some converted multimodal checkpoints expose tokenizer special-token IDs
    that differ from the model generation config.  Fixed actions must retain
    the model's real terminal action rather than being trimmed at a tokenizer-
    only marker.
    """
    generation_config = getattr(model, "generation_config", None)
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        return (int(eos),)
    return tuple(int(value) for value in eos or ())


class HFDecoderCollectorBackend:
    reasoning_prefill = False
    context_prefix_tokens = 0

    def __init__(self, *, model, processor, checkpoint: str, model_family: str, device: str):
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        eos_ids = generation_eos_token_ids(model, self.tokenizer)
        if not eos_ids or self.tokenizer.pad_token_id is None:
            raise ValueError(f"{model_family} collector requires explicit EOS and padding IDs")
        self.identity = CollectorIdentity(
            model_family=model_family,
            tokenizer_fingerprint=fingerprint_tokenizer(self.tokenizer),
            processor_template_fingerprint=fingerprint_processor_template(processor),
            model_config_fingerprint=fingerprint_model_config(checkpoint),
            policy_id=model_state_fingerprint(model),
            eos_token_ids=eos_ids,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )

    def _load_image(self, value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"collector image does not exist: {path}")
        return Image.open(path).convert("RGB")

    def render_prompt(self, instruction: str) -> str:
        raise NotImplementedError

    def prepare_inputs(self, prompt: str, image: Image.Image) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def prepare_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        instruction = sample.get("instruction", sample.get("prompt"))
        if not str(instruction or "").strip():
            raise ValueError("collector sample requires a nonempty instruction/prompt")
        image = self._load_image(sample["image"])
        prompt = self.render_prompt(str(instruction))
        inputs = self.prepare_inputs(prompt, image)
        return {
            "prompt": prompt,
            "image": image,
            "inputs": inputs,
            "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
            "image_hash": multimodal_content_hash({"images": [image], "videos": None}),
            "tokenize_special": lambda text: self.tokenizer.encode(text, add_special_tokens=False),
            "decode_tokens": lambda ids: self.tokenizer.decode(
                list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            "reasoning_prefill": self.reasoning_prefill,
        }

    def _validate_context_length(self, input_length: int, max_new_tokens: int) -> None:
        configs = [getattr(self.model, "config", None)]
        for name in ("text_config", "language_config"):
            nested = getattr(configs[0], name, None) if configs[0] is not None else None
            if nested is not None:
                configs.append(nested)
        positional_limits = []
        fallback_limits = []
        for config in configs:
            for name in ("max_position_embeddings", "n_positions"):
                value = getattr(config, name, None) if config is not None else None
                if isinstance(value, int) and value > 0:
                    positional_limits.append(value)
            for name in ("max_sequence_length", "seq_length", "model_max_length"):
                value = getattr(config, name, None) if config is not None else None
                if isinstance(value, int) and value > 0:
                    fallback_limits.append(value)
        limits = positional_limits or fallback_limits
        requested = input_length + int(self.context_prefix_tokens) + max_new_tokens
        if limits and requested > min(limits):
            raise ValueError(
                f"requested prompt/prefix ({input_length + int(self.context_prefix_tokens)}) + response ({max_new_tokens}) exceeds "
                f"the checkpoint context limit ({min(limits)}); refusing to truncate fixed actions"
            )

    @contextmanager
    def _zero_token_embedding(self, token_index: int):
        embedding = self.model.get_input_embeddings()
        applied = False

        def zero_position(_module, inputs, output):
            nonlocal applied
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return output
            token_ids = inputs[0]
            if token_ids.ndim < 2 or token_ids.shape[-1] <= token_index:
                return output
            masked = output.clone()
            masked[..., token_index, :] = 0
            applied = True
            return masked

        handle = embedding.register_forward_hook(zero_position)
        try:
            yield lambda: applied
        finally:
            handle.remove()

    @torch.no_grad()
    def generate_actions(self, prepared, *, num_samples, max_new_tokens, prefix_ids=(), zero_mask_index=None):
        inputs = {key: value.to(self.device) for key, value in prepared["inputs"].items()}
        prefix = torch.as_tensor(prefix_ids, dtype=torch.long, device=self.device)
        if zero_mask_index is not None:
            if zero_mask_index < 0 or zero_mask_index >= prefix.numel():
                raise ValueError("necessity zero-mask index is outside the supplied reasoning prefix")
        if prefix.numel():
            prefix = prefix.unsqueeze(0)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(prefix)], dim=1
            )
        input_length = inputs["input_ids"].shape[1]
        self._validate_context_length(input_length, max_new_tokens)
        sampling = {"do_sample": False}
        if num_samples > 1:
            sampling = {"do_sample": True, "temperature": 1.0, "top_p": 0.9, "top_k": 0}
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_samples,
            "pad_token_id": self.identity.pad_token_id,
            "eos_token_id": list(self.identity.eos_token_ids),
            **sampling,
        }
        if zero_mask_index is None:
            output = self.model.generate(**generate_kwargs)
        else:
            absolute_index = input_length - prefix.shape[1] + zero_mask_index
            with self._zero_token_embedding(absolute_index) as was_applied:
                output = self.model.generate(**generate_kwargs)
            if not was_applied():
                raise RuntimeError(
                    "necessity intervention did not pass through the model input embedding; "
                    "refusing to substitute a token ID for zero-masking"
                )
        sequences = output.sequences if hasattr(output, "sequences") else output
        continuations = sequences[:, input_length:]
        results = []
        eos = set(self.identity.eos_token_ids)
        for row in continuations:
            ids = row.detach().cpu().tolist()
            stop = next((index for index, token in enumerate(ids) if token in eos), None)
            if stop is not None:
                ids = ids[: stop + 1]
                reason = "eos"
            else:
                while ids and ids[-1] == self.identity.pad_token_id:
                    ids.pop()
                reason = "length"
            results.append(GeneratedAction(tuple(ids), reason))
        return results

    def action_log_probs(self, prepared, action_ids: Sequence[int]) -> torch.Tensor:
        inputs = {key: value.to(self.device) for key, value in prepared["inputs"].items()}
        prompt_length = inputs["input_ids"].shape[1]
        action = torch.as_tensor(action_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        input_ids = torch.cat([inputs["input_ids"], action], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(action)], dim=1)
        forward = dict(inputs)
        forward["input_ids"] = input_ids
        forward["attention_mask"] = attention_mask
        with torch.no_grad():
            logits = self.model(**forward, return_dict=True).logits.float()
        selected = torch.log_softmax(logits[:, :-1], dim=-1).gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return selected[0, prompt_length - 1:prompt_length - 1 + action.shape[1]].cpu()

    def decode_action(self, action_ids):
        return self.tokenizer.decode(list(action_ids), skip_special_tokens=True)
