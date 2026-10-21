"""LLaVA-1.5 fixed-trajectory collector."""

from __future__ import annotations

from transformers import AutoProcessor, LlavaForConditionalGeneration

from .hf_decoder import HFDecoderCollectorBackend, load_peft_adapter, require_local_checkpoint


class Llava15CollectorBackend(HFDecoderCollectorBackend):
    reasoning_prefill = True

    def __init__(self, checkpoint: str, device: str = "cuda", trainable_state: str | None = None, **kwargs):
        checkpoint = require_local_checkpoint(checkpoint, "LLaVA-1.5 checkpoint")
        # Match native verl's processor construction exactly.  The fast and
        # slow CLIP processors have distinct preprocessing identities and a
        # v3 store collected with one must not be consumed with the other.
        processor = AutoProcessor.from_pretrained(checkpoint, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto")
        model = load_peft_adapter(model, trainable_state)
        super().__init__(model=model, processor=processor, checkpoint=checkpoint, model_family="llava15", device=device)

    def render_prompt(self, instruction: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        if not hasattr(self.processor, "apply_chat_template"):
            raise ValueError("LLaVA processor has no chat template; collector refuses to guess one")
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt + " <think>"

    def prepare_inputs(self, prompt, image):
        return self.processor(text=prompt, images=image, return_tensors="pt")
