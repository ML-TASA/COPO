"""Reward backend interface and the COPO outcome reward."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from verl.utils.reward_score.outcome import compute_score


class RewardBackend(ABC):
    """Interface: score generated responses with scalar outcome rewards.

    :meth:`compute` takes the generated responses and, when the reward needs a
    ground truth, the matching answers; it returns one scalar per response.
    These scalars feed the GRPO group-relative advantage; the token-level
    causal reward is added on top by the causal-reward provider.
    """

    @abstractmethod
    def compute(self, responses: list[str], answers: list[object] | None = None,
                tasks: list[str | None] | None = None) -> torch.Tensor:
        """Return a ``(G,)`` float tensor of rewards for ``G`` responses."""


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

    def compute(self, responses: list[str], answers: list[object] | None = None,
                tasks: list[str | None] | None = None) -> torch.Tensor:
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

    ``reward = min(n_tokens, max_tokens) / max_tokens`` (0 for near-empty
    responses).  This is only a fallback for quick smoke tests without
    ground-truth answers; real COPO experiments use :class:`OutcomeReward`.
    The COPO causal reward is added on top regardless.
    """

    def __init__(self, min_tokens: int = 2, max_tokens: int = 64):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def compute(self, responses: list[str], answers: list[object] | None = None,
                tasks: list[str | None] | None = None) -> torch.Tensor:
        values = []
        for text in responses:
            n = len(text.split())
            if n < self.min_tokens:
                values.append(0.0)
            else:
                values.append(min(n, self.max_tokens) / self.max_tokens)
        return torch.tensor(values, dtype=torch.float32)
