"""Reward backend interface and the COPO outcome reward for the MiniGPT-4 loop."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from verl.utils.reward_score.outcome import compute_score


class RewardBackend(ABC):
    """Map generated responses to scalar outcome rewards.

    :meth:`compute` takes the generated responses and, when the reward needs a
    ground truth, the matching answers; it returns a ``(B,)`` float tensor, one
    scalar reward per sampled response.  These scalars feed the GRPO
    group-relative advantage; the token-level causal reward is added on top by
    the causal-reward provider.
    """

    @abstractmethod
    def compute(self, responses, answers=None, tasks=None) -> torch.Tensor:
        """Return rewards of shape ``(B,)`` for ``B`` responses."""


class OutcomeReward(RewardBackend):
    """COPO outcome reward: match each response against its ground-truth answer.

    Delegates to ``verl.utils.reward_score.outcome.compute_score`` — the same
    scorer used to build the offline causal-reward store — so the online
    outcome reward and the offline outcome signal are identical.  This is the
    default COPO outcome reward for real experiments; the token-level causal
    reward is added on top regardless.
    """

    def __init__(self, normalization: str = "strict", task: str | None = None):
        self.normalization = normalization
        self.task = task

    def compute(self, responses, answers=None, tasks=None) -> torch.Tensor:
        if answers is None:
            raise ValueError("OutcomeReward needs ground-truth answers; call compute(responses, answers)")
        if len(answers) != len(responses):
            raise ValueError(f"number of answers ({len(answers)}) != responses ({len(responses)})")
        if tasks is not None and len(tasks) != len(responses):
            raise ValueError(f"number of tasks ({len(tasks)}) != responses ({len(responses)})")
        effective_tasks = tasks if tasks is not None else [self.task] * len(responses)
        scores = [
            float(compute_score(text, answer, normalization=self.normalization, task=task))
            for text, answer, task in zip(responses, answers, effective_tasks)
        ]
        return torch.tensor(scores, dtype=torch.float32)


class DummyReward(RewardBackend):
    """Length reward for smoke tests / interface demos — NOT the paper's outcome reward.

    Rewards a response for being close to a target word count, following a
    Gaussian centered on ``target_words``:

        reward = exp(-0.5 * ((n_words - target) / sigma) ** 2)

    which is ``1.0`` at the target and decays for both shorter and longer
    responses.  This is only a fallback for quick smoke tests without
    ground-truth answers; real COPO experiments use :class:`OutcomeReward`.
    The COPO causal reward is added on top regardless.
    """

    def __init__(self, target_words: int = 16, sigma: float = 12.0):
        self.target_words = float(target_words)
        self.sigma = float(sigma)

    def compute(self, responses, answers=None, tasks=None) -> torch.Tensor:
        n_words = torch.tensor(
            [len(r.split()) for r in responses], dtype=torch.float32
        )
        reward = torch.exp(
            -0.5 * ((n_words - self.target_words) / self.sigma) ** 2
        )
        return reward
