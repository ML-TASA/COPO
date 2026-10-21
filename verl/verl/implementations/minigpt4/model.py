"""Model construction, freeze policy, generation and response log-prob extraction for the MiniGPT-4 backend."""

from __future__ import annotations

import torch

from .config import MiniGPT4Config
from .mini_gpt4 import MiniGPT4


def load_model(config: MiniGPT4Config) -> MiniGPT4:
    """Construct the ``MiniGPT4`` model from ``config``."""
    return MiniGPT4.from_config(config.as_model_kwargs())


def apply_freeze(model: MiniGPT4, config: MiniGPT4Config) -> None:
    """Freeze components per config.

    Official default (``freeze_vit=True``, ``freeze_qformer=True``,
    ``freeze_language_model=True``): train only the ``llama_proj`` projection.
    The vision/Q-Former/LLM freeze flags were already applied inside
    ``MiniGPT4.__init__``; this re-asserts them and adds the projection switch.
    """
    if config.freeze_vit:
        model.visual_encoder.requires_grad_(False)
        model.ln_vision.requires_grad_(False)
    if config.freeze_qformer:
        model.Qformer.requires_grad_(False)
        model.query_tokens.requires_grad_(False)
    if config.freeze_language_model:
        model.llama_model.requires_grad_(False)
    if config.freeze_projection:
        model.llama_proj.requires_grad_(False)


def trainable_parameters(model: MiniGPT4):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model: MiniGPT4, config: MiniGPT4Config):
    """AdamW with the LAVIS decay/no-decay split (decay only on >1-d, non-bias)."""
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


@torch.no_grad()
def generate(model: MiniGPT4, config: MiniGPT4Config, processor, image, instruction: str):
    """Single-sample inference via the official ``MiniGPT4.generate`` path.

    Returns the decoded response string (the ``###Human: … ###Assistant:``
    prompt is stripped by the official post-processing).
    """
    samples = processor.build_sample(image, instruction)
    samples["image"] = samples["image"].to(model.device)
    texts = model.generate(
        samples,
        use_nucleus_sampling=config.use_nucleus_sampling,
        num_beams=config.num_beams,
        max_new_tokens=config.max_new_tokens,
        top_p=config.top_p,
        temperature=config.temperature,
        num_captions=1,
    )
    return texts[0]


@torch.no_grad()
def generate_group(model: MiniGPT4, config: MiniGPT4Config, processor, image, instruction: str, group_size: int):
    """Sample ``group_size`` diverse responses for one ``(image, instruction)``.

    Reuses the official ``MiniGPT4.generate`` path but forces nucleus sampling
    and passes ``num_captions=group_size`` (HF ``num_return_sequences``) so a
    single prompt yields a group of independent samples — the rollout that GRPO
    needs.  Returns a ``list[str]`` of length ``group_size``.
    """
    samples = processor.build_sample(image, instruction)
    samples["image"] = samples["image"].to(model.device)
    texts = model.generate(
        samples,
        use_nucleus_sampling=True,   # diversity is required for the GRPO group
        num_beams=1,
        max_new_tokens=config.grpo_max_new_tokens,
        top_p=config.top_p,
        temperature=config.temperature,
        num_captions=group_size,
    )
    return texts


@torch.no_grad()
def generate_group_with_ids(model: MiniGPT4, config: MiniGPT4Config, processor, image, instruction: str,
                            group_size: int):
    """Return decoded text and the exact sampled action IDs without retokenizing."""
    samples = processor.build_sample(image, instruction)
    samples["image"] = samples["image"].to(model.device)
    texts, response_ids = model.generate(
        samples,
        use_nucleus_sampling=True,
        num_beams=1,
        max_new_tokens=config.grpo_max_new_tokens,
        top_p=config.top_p,
        temperature=config.temperature,
        num_captions=group_size,
        return_token_ids=True,
    )
    return response_ids, texts


def fixed_prompt_token_ids(model: MiniGPT4, prompt: str) -> torch.Tensor:
    """Portable textual prompt identity for MiniGPT-4's image-embedding prompt."""
    before, after = prompt.split("<ImageHere>")
    before_ids = model.llama_tokenizer(before, add_special_tokens=False).input_ids
    after_ids = model.llama_tokenizer(after, add_special_tokens=False).input_ids
    return torch.tensor(
        [int(model.llama_tokenizer.bos_token_id), *map(int, before_ids), *map(int, after_ids)],
        dtype=torch.long,
    )


def decode_response_token_ids(model: MiniGPT4, response_ids: list[torch.Tensor]) -> list[str]:
    """Decode exact actions for scoring without mutating or retokenizing them."""
    decoded = []
    for token_ids in response_ids:
        copy = torch.as_tensor(token_ids).detach().cpu().clone()
        copy[(copy == 0) | (copy == 1)] = 2
        text = model.llama_tokenizer.decode(copy.tolist(), skip_special_tokens=True)
        decoded.append(text.split("###")[0].split("Assistant:")[-1].strip())
    return decoded


def _build_inputs_and_targets(model: MiniGPT4, image, instruction: str, answers: list[str]):
    """Reconstruct the ``inputs_embeds / attention_mask / targets`` triplet that
    ``MiniGPT4.forward`` builds for SFT.
    """
    model.llama_tokenizer.padding_side = "right"
    img_embeds, atts_img = model.encode_img(image)
    img_embeds, atts_img, img_start_pos = model.prompt_wrap(img_embeds, atts_img, instruction)

    text = [t + model.end_sym for t in answers]
    to_regress_tokens = model.llama_tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        add_special_tokens=False,
    ).to(image.device)
    response_lengths = to_regress_tokens.attention_mask.sum(dim=1)
    if torch.any(response_lengths > model.max_txt_len):
        rows = torch.where(response_lengths > model.max_txt_len)[0].tolist()
        raise ValueError(
            f"MiniGPT-4 responses exceed max_txt_len={model.max_txt_len} for rows {rows}; "
            "refusing to truncate because that would break causal-reward alignment"
        )

    batch_size = img_embeds.shape[0]
    bos = torch.ones(
        [batch_size, 1],
        dtype=to_regress_tokens.input_ids.dtype,
        device=to_regress_tokens.input_ids.device,
    ) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.embed_tokens(bos)
    atts_bos = atts_img[:, :1]

    to_regress_embeds = model.embed_tokens(to_regress_tokens.input_ids)
    inputs_embeds, attention_mask, input_lens = model.concat_emb_input_output(
        img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask
    )
    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, attention_mask], dim=1)
    max_positions = getattr(model.llama_model.config, "max_position_embeddings", None)
    if max_positions is not None and inputs_embeds.shape[1] > int(max_positions):
        raise ValueError(
            f"MiniGPT-4 input length {inputs_embeds.shape[1]} exceeds the language model "
            f"context window {max_positions}"
        )

    part_targets = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == model.llama_tokenizer.pad_token_id, -100
    )
    targets = (
        torch.ones(
            [inputs_embeds.shape[0], inputs_embeds.shape[1]], dtype=torch.long
        )
        .to(image.device)
        .fill_(-100)
    )
    for i, target in enumerate(part_targets):
        targets[i, input_lens[i] + 1 : input_lens[i] + len(target) + 1] = target  # +1 for bos
    return inputs_embeds, attention_mask, targets, input_lens


def response_log_probs(model: MiniGPT4, image, instruction: str, answers: list[str]):
    """Gradient-tracking token-level ``log p(response | image, instruction)``.

    Returns ``(log_probs, response_mask, response_token_ids)`` of shape ``(B, L)``
    over the response tokens only. The IDs are used to verify that an offline
    causal-reward row labels the same trajectory.
    """
    inputs_embeds, attention_mask, targets, input_lens = _build_inputs_and_targets(
        model, image, instruction, answers
    )
    with model.maybe_autocast():
        outputs = model.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
    logits = outputs.logits.float()  # (B, T, V)
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)
    shifted = log_probs[:, :-1, :]  # (B, T-1, V)
    # targets were built with the +1 (bos) offset already, so use them directly.
    response_mask = targets != -100  # (B, T)
    shifted_targets = targets[:, 1:]  # (B, T-1)
    shifted_mask = response_mask[:, 1:]  # (B, T-1)
    # ``gather`` indexes into the vocab dim (size V), so the -100 ignore tokens
    # would be out of bounds — clamp them and zero the masked positions after.
    gather_idx = shifted_targets.masked_fill(~shifted_mask, 0)
    gathered = shifted.gather(dim=-1, index=gather_idx.unsqueeze(-1)).squeeze(-1)
    gathered = gathered.masked_fill(~shifted_mask, 0.0)
    # Slice the response span so the returned log-probs / mask are response-only
    # ``(B, L)`` and align 1:1 with the offline causal-reward store.
    prefix_len = int(input_lens[0])
    response_logp = gathered[:, prefix_len:]
    response_only_mask = shifted_mask[:, prefix_len:]
    response_token_ids = shifted_targets[:, prefix_len:].masked_fill(~response_only_mask, 0)
    return response_logp, response_only_mask, response_token_ids


def _build_inputs_and_targets_from_ids(
    model: MiniGPT4,
    image: torch.Tensor,
    instruction: str,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
):
    """Teacher-force exact sampled action IDs; no decode/retokenize/end_sym path."""
    response_ids = torch.as_tensor(response_token_ids, device=image.device).long()
    mask = torch.as_tensor(response_mask, device=image.device).bool()
    if response_ids.ndim != 2 or mask.shape != response_ids.shape:
        raise ValueError("MiniGPT-4 fixed response IDs/mask must be matching 2D tensors")
    lengths = mask.sum(dim=-1)
    if torch.any(lengths == 0) or torch.any(lengths > model.max_txt_len):
        raise ValueError("MiniGPT-4 fixed responses have invalid action lengths")
    expected = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ValueError("MiniGPT-4 fixed response mask must be a contiguous action prefix")

    model.llama_tokenizer.padding_side = "right"
    img_embeds, atts_img = model.encode_img(image)
    img_embeds, atts_img, _ = model.prompt_wrap(img_embeds, atts_img, instruction)
    bos = torch.full(
        (img_embeds.shape[0], 1),
        int(model.llama_tokenizer.bos_token_id),
        dtype=torch.long,
        device=image.device,
    )
    bos_embeds = model.embed_tokens(bos)
    response_embeds = model.embed_tokens(response_ids)
    inputs_embeds, attention_mask, input_lens = model.concat_emb_input_output(
        img_embeds, atts_img, response_embeds, mask.long()
    )
    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_img[:, :1], attention_mask], dim=1)
    max_positions = getattr(model.llama_model.config, "max_position_embeddings", None)
    if max_positions is not None and inputs_embeds.shape[1] > int(max_positions):
        raise ValueError(
            f"MiniGPT-4 input length {inputs_embeds.shape[1]} exceeds the language model context window {max_positions}"
        )
    if len({int(value) for value in input_lens}) != 1:
        raise ValueError("MiniGPT-4 fixed group unexpectedly has unequal prompt lengths")

    targets = torch.full(
        (inputs_embeds.shape[0], inputs_embeds.shape[1]), -100, dtype=torch.long, device=image.device
    )
    part_targets = response_ids.masked_fill(~mask, -100)
    for row, input_len in enumerate(input_lens):
        start = int(input_len) + 1
        targets[row, start:start + response_ids.shape[1]] = part_targets[row]
    return inputs_embeds, attention_mask, targets, input_lens


def response_log_probs_from_ids(
    model: MiniGPT4,
    image: torch.Tensor,
    instruction: str,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
):
    """Current-policy log-probabilities for the exact fixed trajectory actions."""
    inputs_embeds, attention_mask, targets, input_lens = _build_inputs_and_targets_from_ids(
        model, image, instruction, response_token_ids, response_mask
    )
    with model.maybe_autocast():
        outputs = model.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
    logits = outputs.logits.float()
    shifted_targets = targets[:, 1:]
    shifted_mask = shifted_targets != -100
    gather_idx = shifted_targets.masked_fill(~shifted_mask, 0)
    gathered = torch.log_softmax(logits[:, :-1], dim=-1).gather(
        dim=-1, index=gather_idx.unsqueeze(-1)
    ).squeeze(-1)
    prefix_len = int(input_lens[0])
    width = response_token_ids.shape[1]
    log_probs = gathered[:, prefix_len:prefix_len + width].masked_fill(~response_mask.bool(), 0.0)
    actual_mask = shifted_mask[:, prefix_len:prefix_len + width]
    if not torch.equal(actual_mask, response_mask.bool()):
        raise ValueError("MiniGPT-4 teacher-forced action mask changed during label shifting")
    return log_probs, actual_mask
