"""InstructBLIP fixed-trajectory collector."""

from __future__ import annotations

import torch

from verl.implementations.instructblip.config import InstructBlipConfig
from verl.implementations.instructblip.model import apply_freeze, load_model, response_log_probs
from verl.implementations.instructblip.processor import (
    InstructBlipProcessor,
    ORIGINAL_PROTOCOL,
    protocol_processor_fingerprint,
    render_prompt_context,
)
from verl.utils.causal_reward.checkpoint import load_trainable_checkpoint
from verl.utils.causal_reward.collector import CollectorIdentity
from verl.utils.causal_reward.trajectory import multimodal_content_hash

from .hf_decoder import HFDecoderCollectorBackend, require_local_checkpoint


class InstructBlipCollectorBackend(HFDecoderCollectorBackend):
    reasoning_prefill = True

    def __init__(self, checkpoint: str, device: str = "cuda", max_response_length: int = 2048,
                 trainable_state: str | None = None,
                 generation_protocol: str = ORIGINAL_PROTOCOL, **kwargs):
        checkpoint = require_local_checkpoint(checkpoint, "InstructBLIP checkpoint")
        config = InstructBlipConfig(
            model_name_or_path=checkpoint,
            device=device,
            max_txt_len=max_response_length,
            dtype=kwargs.get("dtype", "float32"),
            generation_protocol=generation_protocol,
        )
        config.validate()
        model = load_model(config)
        apply_freeze(model, config)
        if trainable_state:
            load_trainable_checkpoint(
                trainable_state, model, expected_model_family="instructblip"
            )
        if bool(getattr(model.language_model.config, "is_encoder_decoder", False)):
            raise ValueError(
                "fixed InstructBLIP collection currently requires the paper's decoder-only/Vicuna variant"
            )
        processor = InstructBlipProcessor(config)
        super().__init__(model=model, processor=processor, checkpoint=checkpoint, model_family="instructblip", device=device)
        self.generation_protocol = generation_protocol
        self.context_prefix_tokens = int(model.query_tokens.shape[1])
        self.identity = CollectorIdentity(
            **{
                **self.identity.__dict__,
                "processor_template_fingerprint": protocol_processor_fingerprint(
                    processor._hf, generation_protocol
                ),
            }
        )

    def render_prompt(self, instruction: str) -> str:
        return render_prompt_context(
            instruction, getattr(self, "generation_protocol", ORIGINAL_PROTOCOL)
        )[0]

    def prepare_inputs(self, prompt, image, *, qformer_prompt=None):
        return self.processor.generate_inputs(
            image,
            prompt,
            qformer_instruction=qformer_prompt,
        )

    def prepare_sample(self, sample):
        """Build the exact split or Original contexts without altering actions."""
        instruction = sample.get("instruction", sample.get("prompt"))
        if not str(instruction or "").strip():
            raise ValueError("collector sample requires a nonempty instruction/prompt")
        image = self._load_image(sample["image"])
        prompt, qformer_prompt = render_prompt_context(
            str(instruction), self.generation_protocol
        )
        inputs = self.prepare_inputs(prompt, image, qformer_prompt=qformer_prompt)
        return {
            "prompt": prompt,
            "qformer_prompt": qformer_prompt,
            "image": image,
            "inputs": inputs,
            "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
            "image_hash": multimodal_content_hash({"images": [image], "videos": None}),
            "tokenize_special": lambda text: self.tokenizer.encode(
                text, add_special_tokens=False
            ),
            "decode_tokens": lambda ids: self.tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "reasoning_prefill": self.reasoning_prefill,
            "protocol_id": self.generation_protocol,
        }

    def decode_action(self, action_ids):
        return self.processor.decode(list(action_ids))

    def action_log_probs(self, prepared, action_ids):
        """Score the exact action through the training-side response alignment."""
        sample = self.processor.process_response_ids(
            prepared["image"],
            prepared["prompt"],
            action_ids,
            qformer_instruction=prepared["qformer_prompt"],
        )
        batch = {
            key: value.to(self.device)
            for key, value in self.processor.collate([sample]).items()
        }
        with torch.no_grad():
            log_probs, mask = response_log_probs(
                self.model, batch, response_length=len(action_ids)
            )
        expected = torch.ones_like(mask, dtype=torch.bool)
        if not torch.equal(mask, expected):
            raise ValueError("InstructBLIP response-only scoring lost an action token")
        return log_probs[0].detach().cpu()
