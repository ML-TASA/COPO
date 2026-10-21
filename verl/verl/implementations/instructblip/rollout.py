"""Multi-response rollout via the official HF ``generate`` path."""

from __future__ import annotations

import torch


def _generation_eos_id(model, tokenizer):
    generation_config = getattr(model, "generation_config", None)
    eos_id = getattr(generation_config, "eos_token_id", None)
    return tokenizer.eos_token_id if eos_id is None else eos_id


def _trim_response(ids: list[int], eos_id, pad_id, *, keep_eos: bool = False) -> list[int]:
    """Trim a continuation, optionally retaining the sampled EOS action."""
    if eos_id is not None and eos_id in ids:
        stop = ids.index(eos_id)
        ids = ids[: stop + 1] if keep_eos else ids[:stop]
    while ids and pad_id is not None and ids[-1] == pad_id:
        ids.pop()
    return ids


def _generated_continuation(output_ids, input_length: int, is_encoder_decoder: bool):
    """Strip the prompt only for decoder-only language models."""
    return output_ids if is_encoder_decoder else output_ids[input_length:]


@torch.no_grad()
def generate_responses(model, processor, image, instruction: str, group_size: int,
                       max_new_tokens: int = 2048, temperature: float = 1.0,
                       top_p: float = 0.9, do_sample: bool = True,
                       keep_eos: bool = False):
    """Sample ``group_size`` responses for one (image, instruction) prompt.

    Returns ``(response_ids, texts)`` where ``response_ids`` is a list of ``(T,)``
    long tensors (the continuation after the instruction prefix) and ``texts``
    the decoded strings.
    """
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    inputs = processor.generate_inputs(image, instruction)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # Rollout must run in eval mode (dropout off) and never leak the training-mode
    # state into the sampled sequences.
    was_training = model.training
    model.eval()
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
        )
    finally:
        model.train(was_training)

    eos_id = _generation_eos_id(model, tok)
    if isinstance(eos_id, (list, tuple)):
        if len(eos_id) != 1:
            raise ValueError("InstructBLIP rollout currently requires one generation EOS token ID")
        eos_id = eos_id[0]
    is_encoder_decoder = bool(getattr(model.language_model.config, "is_encoder_decoder", False))
    response_ids, texts = [], []
    for i in range(outputs.shape[0]):
        continuation = _generated_continuation(outputs[i], input_len, is_encoder_decoder)
        trimmed = _trim_response(continuation.cpu().tolist(), eos_id, pad_id, keep_eos=keep_eos)
        t = torch.tensor(trimmed, dtype=torch.long, device=model.device)
        response_ids.append(t)
        texts.append(processor.decode(trimmed))
    return response_ids, texts
