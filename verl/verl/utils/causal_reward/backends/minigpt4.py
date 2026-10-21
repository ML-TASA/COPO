"""MiniGPT-4 fixed-trajectory collector using exact action embeddings."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from verl.implementations.minigpt4.config import MiniGPT4Config
from verl.implementations.minigpt4.model import (
    apply_freeze,
    decode_response_token_ids,
    fixed_prompt_token_ids,
    generate_group_with_ids,
    load_model,
    response_log_probs_from_ids,
)
from verl.implementations.minigpt4.processor import MiniGPT4Processor
from verl.utils.causal_reward.collector import CollectorIdentity, GeneratedAction, model_state_fingerprint
from verl.utils.causal_reward.checkpoint import load_trainable_checkpoint
from verl.utils.causal_reward.trajectory import (
    fingerprint_mapping,
    fingerprint_model_config,
    fingerprint_tokenizer,
    multimodal_content_hash,
)


class MiniGPT4CollectorBackend:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        *,
        llama_model: str,
        vit_pretrained_path: str,
        qformer_model: str,
        bert_model_path: str,
        max_response_length: int = 2048,
        trainable_state: str | None = None,
        **kwargs,
    ):
        config = MiniGPT4Config(
            llama_model=llama_model,
            vit_pretrained_path=vit_pretrained_path,
            q_former_model=qformer_model,
            ckpt=checkpoint,
            bert_model_path=bert_model_path,
            device=device,
            dtype=kwargs.get("dtype", "float16"),
            max_txt_len=max_response_length,
            grpo_max_new_tokens=max_response_length,
        )
        config.validate()
        self.config = config
        self.model = load_model(config).to(config.resolve_device()).eval()
        apply_freeze(self.model, config)
        if trainable_state:
            load_trainable_checkpoint(
                trainable_state, self.model, expected_model_family="minigpt4"
            )
        self.processor = MiniGPT4Processor(config)
        self.tokenizer = self.model.llama_tokenizer
        eos = self.tokenizer.eos_token_id
        eos_ids = (int(eos),) if isinstance(eos, int) else tuple(int(value) for value in eos or ())
        if not eos_ids or self.tokenizer.pad_token_id is None:
            raise ValueError("MiniGPT-4 collector requires explicit EOS and padding IDs")
        processor_fp = fingerprint_mapping({
            "class": f"{type(self.processor).__module__}.{type(self.processor).__qualname__}",
            "prompt_template": config.prompt_template,
            "image_size": config.image_size,
        })
        model_fp = fingerprint_mapping({
            "llama_config": fingerprint_model_config(config.llama_model),
            "vit_model": config.vit_model,
            "image_size": config.image_size,
            "num_query_token": config.num_query_token,
            "has_qformer": config.has_qformer,
            "prompt_template": config.prompt_template,
            "end_sym": config.end_sym,
        })
        self.identity = CollectorIdentity(
            model_family="minigpt4",
            tokenizer_fingerprint=fingerprint_tokenizer(self.tokenizer),
            processor_template_fingerprint=processor_fp,
            model_config_fingerprint=model_fp,
            policy_id=model_state_fingerprint(self.model),
            eos_token_ids=eos_ids,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )

    def prepare_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        instruction = sample.get("instruction", sample.get("prompt"))
        if not str(instruction or "").strip():
            raise ValueError("MiniGPT-4 collector requires a nonempty instruction")
        image_value = sample["image"]
        if isinstance(image_value, Image.Image):
            image = image_value.convert("RGB")
        else:
            image = Image.open(str(image_value)).convert("RGB")
        prompt = self.processor.build_prompt(str(instruction))
        return {
            "instruction": str(instruction),
            "prompt": prompt,
            "image": image,
            "image_tensor": self.processor.preprocess_image(image).unsqueeze(0).to(self.model.device),
            "prompt_token_ids": fixed_prompt_token_ids(self.model, prompt).tolist(),
            "image_hash": multimodal_content_hash({"images": [image], "videos": None}),
            "tokenize_special": lambda text: self.tokenizer.encode(text, add_special_tokens=False),
            "decode_tokens": lambda ids: self.tokenizer.decode(
                list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
        }

    @torch.no_grad()
    def generate_actions(self, prepared, *, num_samples, max_new_tokens, prefix_ids=(), zero_mask_index=None):
        language_config = getattr(getattr(self.model, "llama_model", None), "config", None)
        context_limit = getattr(language_config, "max_position_embeddings", None)
        if isinstance(context_limit, int):
            prompt_length = len(prepared["prompt_token_ids"]) + len(prefix_ids)
            if prompt_length + max_new_tokens > context_limit:
                raise ValueError(
                    f"MiniGPT-4 prompt/prefix + response exceeds context limit {context_limit}; "
                    "refusing to truncate fixed actions"
                )
        if not prefix_ids and zero_mask_index is None:
            generation_config = replace(self.config, grpo_max_new_tokens=max_new_tokens)
            action_ids, _ = generate_group_with_ids(
                self.model,
                generation_config,
                self.processor,
                prepared["image"],
                prepared["instruction"],
                num_samples,
            )
            eos = set(self.identity.eos_token_ids)
            return [
                GeneratedAction(
                    tuple(int(token) for token in ids.detach().cpu().tolist()),
                    "eos" if len(ids) and int(ids[-1]) in eos else "length",
                )
                for ids in action_ids
            ]
        image = prepared["image_tensor"]
        img_embeds, atts_img = self.model.encode_img(image)
        prompt_embeds, prompt_mask, _ = self.model.prompt_wrap(img_embeds, atts_img, prepared["prompt"])
        bos = torch.full((1, 1), int(self.tokenizer.bos_token_id), device=image.device, dtype=torch.long)
        inputs = torch.cat([self.model.embed_tokens(bos), prompt_embeds], dim=1)
        attention = torch.cat([prompt_mask[:, :1], prompt_mask], dim=1)
        prefix = torch.as_tensor(prefix_ids, device=image.device, dtype=torch.long)
        if prefix.numel():
            prefix_embeds = self.model.embed_tokens(prefix.unsqueeze(0))
            if zero_mask_index is not None:
                if zero_mask_index < 0 or zero_mask_index >= prefix.numel():
                    raise ValueError("MiniGPT-4 necessity index is outside the reasoning prefix")
                prefix_embeds = prefix_embeds.clone()
                prefix_embeds[:, zero_mask_index] = 0
            inputs = torch.cat([inputs, prefix_embeds], dim=1)
            attention = torch.cat([attention, torch.ones(1, prefix.numel(), device=image.device, dtype=attention.dtype)], dim=1)
        sampling = {"do_sample": False}
        if num_samples > 1:
            sampling = {"do_sample": True, "temperature": 1.0, "top_p": 0.9}
        output = self.model.llama_model.generate(
            inputs_embeds=inputs,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_samples,
            **sampling,
        )
        sequences = output.sequences if hasattr(output, "sequences") else output
        results = []
        eos = set(self.identity.eos_token_ids)
        for row in sequences:
            ids = row.detach().cpu().tolist()
            stop = next((index for index, token in enumerate(ids) if token in eos), None)
            if stop is not None:
                ids, reason = ids[: stop + 1], "eos"
            else:
                while ids and ids[-1] == self.identity.pad_token_id:
                    ids.pop()
                reason = "length"
            results.append(GeneratedAction(tuple(ids), reason))
        return results

    def action_log_probs(self, prepared, action_ids: Sequence[int]) -> torch.Tensor:
        ids = torch.as_tensor(action_ids, device=self.model.device, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(ids, dtype=torch.bool)
        log_probs, actual_mask = response_log_probs_from_ids(
            self.model, prepared["image_tensor"], prepared["prompt"], ids, mask
        )
        if not torch.equal(mask, actual_mask):
            raise ValueError("MiniGPT-4 action mask changed during collector scoring")
        return log_probs[0].detach().cpu()

    def decode_action(self, action_ids):
        return decode_response_token_ids(self.model, [torch.as_tensor(action_ids)])[0]
