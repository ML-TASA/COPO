"""Image + instruction + response -> model inputs (official HF tokenization)."""

from __future__ import annotations

import torch
from transformers import InstructBlipProcessor as _TransformersInstructBlipProcessor

from verl.utils.causal_reward.trajectory import fingerprint_mapping, fingerprint_processor_template

from .alignment import append_response_tokens
from .config import InstructBlipConfig


THINK_PREFILL = "<think>"
ORIGINAL_PROTOCOL = "original"
SPLIT_CONTEXT_TWO_SHOT_PROTOCOL = "split_context_two_shot"
SUPPORTED_PROTOCOLS = (ORIGINAL_PROTOCOL, SPLIT_CONTEXT_TWO_SHOT_PROTOCOL)

# The canonical data builder appends this exact suffix for the Original
# protocol.  The optional split-context protocol removes only this known
# suffix before constructing its own format-only context; arbitrary structural
# markup is rejected instead of being guessed or silently rewritten.
ORIGINAL_FORMAT_INSTRUCTION = (
    " Follow this exact response protocol. Example response: "
    "<think>I inspect the relevant visual evidence.</think><answer>the final answer</answer>. "
    "Now provide your own image-grounded reasoning and final answer in the same format; "
    "do not omit or rename any delimiter."
)

_SPLIT_CONTEXT_TEMPLATE_VERSION = "instructblip.split_context_two_shot.v1"


def render_copo_prompt(instruction: str) -> str:
    """Render the prompt-prefilled COPO protocol for HF InstructBLIP."""
    prompt = str(instruction).rstrip()
    if not prompt:
        raise ValueError("InstructBLIP COPO prompt requires a nonempty instruction")
    if "<image>" in prompt:
        raise ValueError(
            "InstructBLIPProcessor inserts image placeholders automatically; "
            "the instruction must not contain a literal <image> token"
        )
    if prompt.endswith(THINK_PREFILL):
        return prompt
    return f"{prompt}\nAssistant: {THINK_PREFILL}"


def _base_instruction(instruction: str) -> str:
    prompt = str(instruction).rstrip()
    if prompt.endswith(ORIGINAL_FORMAT_INSTRUCTION):
        prompt = prompt[: -len(ORIGINAL_FORMAT_INSTRUCTION)].rstrip()
    if not prompt:
        raise ValueError("InstructBLIP COPO prompt requires a nonempty instruction")
    if "<image>" in prompt:
        raise ValueError(
            "InstructBLIPProcessor inserts image placeholders automatically; "
            "the instruction must not contain a literal <image> token"
        )
    return prompt


def render_prompt_context(
    instruction: str,
    protocol: str = ORIGINAL_PROTOCOL,
) -> tuple[str, str]:
    """Return the exact ``(LLM prompt, Q-Former prompt)`` for a protocol.

    Original behavior is intentionally delegated to :func:`render_copo_prompt`.
    The optional split-context path remains a single autoregressive action:
    demonstrations and the opening ``<think>`` are prompt tokens, while the
    generated reasoning, closing delimiters, answer, and EOS are action tokens.
    """
    if protocol == ORIGINAL_PROTOCOL:
        prompt = render_copo_prompt(instruction)
        return prompt, prompt
    if protocol != SPLIT_CONTEXT_TWO_SHOT_PROTOCOL:
        raise ValueError(f"unsupported InstructBLIP generation protocol: {protocol!r}")

    question = _base_instruction(instruction)
    if any(marker in question for marker in ("<think>", "</think>", "<answer>", "</answer>")):
        raise ValueError(
            "split_context_two_shot requires an unformatted task instruction; "
            "only the canonical Original format suffix can be removed safely"
        )
    llm_prompt = (
        "These are format-only examples about unrelated hypothetical images.\n"
        "Example 1 question: Is there a bicycle?\n"
        "Assistant: <think>I inspect the scene and a bicycle is clearly visible."
        "</think><answer>yes</answer>\n"
        "Example 2 question: Is there an elephant?\n"
        "Assistant: <think>I inspect the scene and no elephant is visible."
        "</think><answer>no</answer>\n"
        f"Current question: {question}\n"
        "Use only the current image. Give brief visual evidence and then answer yes or no "
        "in exactly the demonstrated format.\n"
        f"Assistant: {THINK_PREFILL}"
    )
    return llm_prompt, question


def protocol_processor_fingerprint(hf_processor, protocol: str) -> str:
    """Fingerprint both the HF processor and the selected prompt protocol."""
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported InstructBLIP generation protocol: {protocol!r}")
    base = fingerprint_processor_template(hf_processor)
    # Preserve the identity of every existing Original v3 store exactly.
    if protocol == ORIGINAL_PROTOCOL:
        return base
    return fingerprint_mapping(
        {
            "hf_processor": base,
            "generation_protocol": protocol,
            "template_version": _SPLIT_CONTEXT_TEMPLATE_VERSION,
        }
    )


class InstructBlipProcessor:
    def __init__(self, config: InstructBlipConfig):
        self.config = config
        self._hf = _TransformersInstructBlipProcessor.from_pretrained(config.model_name_or_path)
        self.tokenizer = self._hf.tokenizer
        self.qformer_tokenizer = self._hf.qformer_tokenizer

    def process(
        self,
        image,
        instruction: str,
        response: str,
        *,
        qformer_instruction: str | None = None,
    ) -> dict:
        """Return a single-sample dict of ``(1, ...)`` tensors (no padding)."""
        full = self._hf(images=image, text=instruction + " " + response, return_tensors="pt")
        instr = self._hf(images=image, text=instruction, return_tensors="pt")
        n_prefix = instr["input_ids"].shape[1]

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = input_ids.clone()
        labels[:, :n_prefix] = -100  # mask image tokens + BOS + instruction

        qformer = self.qformer_tokenizer(
            qformer_instruction if qformer_instruction is not None else instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config.qformer_max_len,
        )

        return {
            "pixel_values": full["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "qformer_input_ids": qformer["input_ids"],
            "qformer_attention_mask": qformer["attention_mask"],
            "labels": labels,
        }

    def process_response_ids(
        self,
        image,
        instruction: str,
        response_ids,
        *,
        qformer_instruction: str | None = None,
    ) -> dict:
        """Build training inputs from the exact token IDs emitted by rollout."""
        prompt = self._hf(images=image, text=instruction, return_tensors="pt")
        input_ids, attention_mask, labels = append_response_tokens(
            prompt["input_ids"],
            prompt["attention_mask"],
            torch.as_tensor(response_ids).detach().cpu(),
            self.config.max_txt_len,
        )
        qformer = self.qformer_tokenizer(
            qformer_instruction if qformer_instruction is not None else instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config.qformer_max_len,
        )
        return {
            "pixel_values": prompt["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "qformer_input_ids": qformer["input_ids"],
            "qformer_attention_mask": qformer["attention_mask"],
            "labels": labels,
        }

    def generate_inputs(
        self,
        image,
        instruction: str,
        *,
        qformer_instruction: str | None = None,
    ) -> dict:
        """Inference inputs (no labels): image tokens + instruction only."""
        full = self._hf(images=image, text=instruction, return_tensors="pt")
        qformer = self.qformer_tokenizer(
            qformer_instruction if qformer_instruction is not None else instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config.qformer_max_len,
        )
        return {
            "pixel_values": full["pixel_values"],
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "qformer_input_ids": qformer["input_ids"],
            "qformer_attention_mask": qformer["attention_mask"],
        }

    def collate(self, samples: list[dict]) -> dict:
        """Pad a list of :meth:`process` outputs into a single batch."""
        batch = {"pixel_values": torch.cat([s["pixel_values"] for s in samples], dim=0)}
        for key, pad_value in (
            ("input_ids", self.tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("labels", -100),
        ):
            batch[key] = self._pad([s[key] for s in samples], pad_value)
        for key, pad_value in (
            ("qformer_input_ids", self.qformer_tokenizer.pad_token_id),
            ("qformer_attention_mask", 0),
        ):
            batch[key] = self._pad([s[key] for s in samples], pad_value)
        return batch

    def decode(self, token_ids) -> str:
        return self._hf.decode(token_ids, skip_special_tokens=True)

    @staticmethod
    def _pad(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[1] < max_len:
                t = torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=pad_value)
            padded.append(t)
        return torch.cat(padded, dim=0)
