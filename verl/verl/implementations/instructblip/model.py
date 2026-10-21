"""Model loading, freeze policy, forward loss and response log-prob extraction."""

from __future__ import annotations

import torch
from transformers import InstructBlipForConditionalGeneration

from .alignment import extract_response_only
from .config import InstructBlipConfig


def load_model(config: InstructBlipConfig) -> InstructBlipForConditionalGeneration:
    """Load the official HF InstructBLIP checkpoint."""
    return InstructBlipForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        dtype=config.to_torch_dtype(),
    )


def apply_freeze(model: InstructBlipForConditionalGeneration, config: InstructBlipConfig) -> None:
    """Freeze components per config.

    Official default (``freeze_vision_encoder=True``, ``freeze_language_model=True``):
    train only the Q-Former, the query tokens and the language projection.
    """
    if config.freeze_vision_encoder:
        model.vision_model.requires_grad_(False)
    if config.freeze_language_model:
        model.language_model.requires_grad_(False)
    if config.freeze_qformer:
        model.qformer.requires_grad_(False)
        model.query_tokens.requires_grad_(False)
    if config.freeze_projection:
        model.language_projection.requires_grad_(False)


def trainable_parameters(model: InstructBlipForConditionalGeneration):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model: InstructBlipForConditionalGeneration, config: InstructBlipConfig):
    """AdamW with the LAVIS decay/no-decay split (weight decay only on >1-d, non-bias params)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr)


def _forward_inputs(batch: dict) -> dict:
    return {
        "pixel_values": batch["pixel_values"],
        "qformer_input_ids": batch["qformer_input_ids"],
        "qformer_attention_mask": batch["qformer_attention_mask"],
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }


def forward_loss(model: InstructBlipForConditionalGeneration, batch: dict) -> torch.Tensor:
    """Cross-entropy loss over response tokens only (``labels`` masks the rest)."""
    outputs = model(**_forward_inputs(batch), labels=batch["labels"], return_dict=True)
    return outputs.loss


def _log_probs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor,
                           response_length: int = 2048):
    """Causal-shifted token log-probs + a response-only mask, from HF logits.

    ``logits`` shape ``(B, T, V)`` over the full LLM sequence (32 visual-prefix
    slots + instruction + response).  logits at position ``t-1`` predict token
    ``t``, so gather ``log_softmax(logits[:, :-1])`` at ``input_ids[:, 1:]``.
    The response tokens (``labels != -100``) are then extracted into a
    response-only ``(B, response_length)`` view aligned to the causal-reward
    store — image / instruction / full-sequence positions are excluded.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # (B, T, V)
    shifted = log_probs[:, :-1, :]  # (B, T-1, V)
    targets = input_ids[:, 1:]  # (B, T-1)
    response_mask = labels[:, 1:] != -100  # (B, T-1)
    gathered = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return extract_response_only(gathered, response_mask, response_length)


def response_log_probs(model: InstructBlipForConditionalGeneration, batch: dict,
                       response_length: int = 2048):
    """Gradient-tracking token log-probs over the response, ``(B, response_length)``.

    Returns ``(log_probs, response_mask)`` over response tokens only, aligned to
    the offline causal-reward store.  Used by the GRPO policy-gradient loss.
    """
    outputs = model(**_forward_inputs(batch), return_dict=True)
    return _log_probs_from_logits(outputs.logits, batch["input_ids"], batch["labels"], response_length)


def extract_response_log_probs(model: InstructBlipForConditionalGeneration, batch: dict,
                               response_length: int = 2048) -> torch.Tensor:
    """No-grad token-level ``log p(response | image, instruction)``, ``(B, response_length)``."""
    with torch.no_grad():
        outputs = model(**_forward_inputs(batch), return_dict=True)
    gathered, response_mask = _log_probs_from_logits(
        outputs.logits, batch["input_ids"], batch["labels"], response_length
    )
    return gathered.masked_fill(~response_mask, 0.0)


@torch.no_grad()
def generate(model: InstructBlipForConditionalGeneration, processor, image, instruction: str, max_new_tokens: int = 2048, **gen_kwargs):
    """Single-sample inference via the official HF ``generate`` path.

    Returns ``(generated_token_ids, decoded_text)`` where ``decoded_text`` is the
    newly generated response (the image/instruction prefix is sliced off).
    """
    inputs = processor.generate_inputs(image, instruction)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    is_encoder_decoder = bool(getattr(model.language_model.config, "is_encoder_decoder", False))
    generated_ids = outputs[0] if is_encoder_decoder else outputs[0][input_len:]
    text = processor.decode(generated_ids.detach().cpu().tolist())
    return generated_ids, text
