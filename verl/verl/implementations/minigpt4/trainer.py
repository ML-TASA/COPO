"""Token-level COPO training for the MiniGPT-4 backend."""

from __future__ import annotations

import torch

from verl.utils.causal_reward.adapter import compute_fixed_copo_advantages, prepare_fixed_adapter_batch
from verl.utils.causal_reward.optimization import compute_copo_eq11_total_loss
from verl.utils.causal_reward.trajectory import multimodal_content_hash

from .config import MiniGPT4Config
from .model import (
    build_optimizer,
    decode_response_token_ids,
    fixed_prompt_token_ids,
    response_log_probs_from_ids,
)
from .reward import RewardBackend


class COPOTrainer:
    """Strict fixed-trajectory Eq. 11 trainer for MiniGPT-4.

    Only ``llama_proj`` is trainable under the default freeze policy.
    """

    def __init__(self, model, config: MiniGPT4Config, processor, outcome_backend: RewardBackend,
                 causal_provider=None, answers=None, tasks=None, sample_ids=None,
                 fixed_store=None, group_ids=None, image_ids=None):
        self.model = model
        self.config = config
        self.processor = processor
        self.outcome_backend = outcome_backend
        self.causal_provider = causal_provider
        self.optimizer = build_optimizer(model, config)
        self.group_size = config.group_size
        self.answers = list(answers) if answers is not None else None
        self.tasks = list(tasks) if tasks is not None else None
        self.sample_ids = list(sample_ids) if sample_ids is not None else None
        self.fixed_store = fixed_store
        self.group_ids = list(group_ids) if group_ids is not None else None
        self.image_ids = list(image_ids) if image_ids is not None else None

    def step(self, images, instructions):
        """Run one strict fixed-trajectory Eq. 11 update."""
        if self.fixed_store is not None:
            return self._fixed_step(images, instructions)
        raise RuntimeError(
            "formal MiniGPT-4 COPO training requires a validated "
            "copo.fixed_trajectory.v3 store; the legacy rollout policy-gradient "
            "path is not the Eq. 11 objective"
        )

    def _fixed_step(self, images, instructions):
        """One strict Eq. 11 update over exact v3 MiniGPT-4 actions."""
        if not all(values is not None for values in (self.sample_ids, self.group_ids, self.image_ids)):
            raise ValueError("fixed MiniGPT-4 training requires sample/group/image identities")
        total_loss = None
        for index, (image, instruction) in enumerate(zip(images, instructions, strict=True)):
            image_hash = multimodal_content_hash({"images": [image], "videos": None})
            selected = self.fixed_store.select_groups(
                [self.sample_ids[index]],
                [self.group_ids[index]],
                [self.image_ids[index]],
                [image_hash],
            )
            stored_ids = selected._tensor("response_token_ids").long().to(self.model.device)
            stored_mask = selected._tensor("response_mask").bool().to(self.model.device)
            preprocessed = self.processor.preprocess_image(image).unsqueeze(0)
            image_batch = preprocessed.repeat(self.group_size, 1, 1, 1).to(self.model.device)
            prompt = self.processor.build_prompt(instruction)
            current_log_probs, current_mask = response_log_probs_from_ids(
                self.model, image_batch, prompt, stored_ids, stored_mask
            )
            fixed = prepare_fixed_adapter_batch(
                selected,
                prompt_token_ids=fixed_prompt_token_ids(self.model, prompt),
                response_token_ids=stored_ids,
                response_mask=current_mask,
                device=current_log_probs.device,
            )

            lengths = selected._tensor("response_lengths").long().tolist()
            exact_actions = [stored_ids[row, : int(length)].detach() for row, length in enumerate(lengths)]
            texts = decode_response_token_ids(self.model, exact_actions)
            answers = [self.answers[index]] * self.group_size
            tasks = [self.tasks[index]] * self.group_size
            checked_outcomes = self.outcome_backend.compute(texts, answers, tasks).to(fixed.outcome_rewards.device)
            if not torch.allclose(checked_outcomes.float(), fixed.outcome_rewards, atol=1e-6, rtol=0):
                raise ValueError("fixed MiniGPT-4 outcome rewards do not match the exact stored actions")

            advantages, _ = compute_fixed_copo_advantages(fixed)
            group_loss, _ = compute_copo_eq11_total_loss(
                fixed.old_log_probs,
                current_log_probs,
                fixed.reference_log_probs,
                advantages,
                fixed.response_mask,
                self.config.clip_ratio,
                self.config.clip_ratio,
                self.config.kl_loss_coef,
            )
            total_loss = group_loss if total_loss is None else total_loss + group_loss

        if total_loss is None:
            raise ValueError("fixed MiniGPT-4 update received no samples")
        loss = total_loss / len(images)
        self.optimizer.zero_grad()
        loss.backward()
        trainable_grads = [
            parameter.grad.detach().norm()
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not trainable_grads:
            raise RuntimeError("fixed MiniGPT-4 update produced no trainable gradients")
        grad_norm = torch.stack(trainable_grads).norm().item()
        self.optimizer.step()
        return {"loss": loss.detach().item(), "grad_norm": grad_norm}
