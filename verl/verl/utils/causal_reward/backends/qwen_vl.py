"""Original Qwen-VL-Chat (2023) fixed-trajectory collector."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from verl.implementations.qwen_vl import QWenVLProcessor, ensure_qwen_vl_tokenizer, is_qwen_vl, wrap_qwen_vl_for_verl
from verl.implementations.qwen_vl.legacy_compat import install_qwen_vl_compat

from .hf_decoder import HFDecoderCollectorBackend, load_peft_adapter, require_local_checkpoint


class QwenVLCollectorBackend(HFDecoderCollectorBackend):
    reasoning_prefill = True

    def __init__(self, checkpoint: str, device: str = "cuda", trainable_state: str | None = None, **kwargs):
        checkpoint = require_local_checkpoint(checkpoint, "Qwen-VL checkpoint")
        install_qwen_vl_compat()
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        if not is_qwen_vl(config):
            raise ValueError("qwenvl collector requires original Qwen-VL-Chat, not Qwen2/2.5-VL")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        ensure_qwen_vl_tokenizer(tokenizer)
        processor = QWenVLProcessor(tokenizer, config)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype="auto")
        wrap_qwen_vl_for_verl(model)
        model = load_peft_adapter(model, trainable_state)
        super().__init__(model=model, processor=processor, checkpoint=checkpoint, model_family="qwenvl", device=device)

    def render_prompt(self, instruction: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt + "<think>"

    def prepare_inputs(self, prompt, image):
        return self.processor(text=[prompt], images=[image], return_tensors="pt")
