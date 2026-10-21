from __future__ import annotations

import copy

import pytest
import torch

from verl.utils.causal_reward.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint
from verl.utils.causal_reward.optimization import compute_copo_eq11_total_loss


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.12, -0.08], [0.04, 0.09]]))
        self.frozen = torch.nn.Parameter(torch.tensor([0.3, -0.2]), requires_grad=False)

    def log_probs(self):
        features = torch.tensor(
            [[[1.0, 0.0], [0.5, 1.0], [1.0, -0.5]], [[0.0, 1.0], [1.0, 0.5], [-0.5, 1.0]]]
        )
        logits = torch.einsum("btd,vd->btv", features, self.weight + self.frozen)
        actions = torch.tensor([[0, 1, 0], [1, 0, 1]])
        return torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _update(initial, causal):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.SGD([model.weight], lr=0.05)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    original = torch.tensor([[0.4, 0.4, 0.0], [-0.4, -0.4, -0.4]])
    advantage = original + causal
    before_log_probs = model.log_probs().detach().clone()
    old = before_log_probs.detach().clone()
    reference = old - torch.tensor([[0.05, -0.02, 0.0], [0.03, -0.04, 0.02]])
    old_before = old.clone()
    reference_before = reference.clone()
    before = model.weight.detach().clone()
    frozen_before = model.frozen.detach().clone()
    loss, _ = compute_copo_eq11_total_loss(
        old, model.log_probs(), reference, advantage, mask, 0.2, 0.2, 0.01
    )
    optimizer.zero_grad()
    loss.backward()
    gradient = model.weight.grad.detach().clone()
    optimizer.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "loss": loss.detach(),
        "gradient": gradient,
        "update": model.weight.detach() - before,
        "log_prob_change": model.log_probs().detach() - before_log_probs,
        "frozen_before": frozen_before,
        "old_unchanged": torch.equal(old, old_before),
        "reference_unchanged": torch.equal(reference, reference_before),
    }


def test_copo_changes_gradient_and_update_only_through_reasoning_reward():
    torch.manual_seed(7)
    initial = _Policy()
    zero = torch.zeros(2, 3)
    causal = torch.tensor([[0.35, 0.0, 0.0], [-0.14, 0.0, 0.0]])
    grpo = _update(initial, zero)
    copo = _update(initial, causal)
    assert torch.isfinite(copo["loss"])
    assert torch.isfinite(copo["gradient"]).all()
    assert copo["gradient"].norm().item() > 0
    assert copo["update"].norm().item() > 0
    assert copo["log_prob_change"].norm().item() > 0
    assert torch.equal(copo["model"].frozen.detach(), copo["frozen_before"])
    assert copo["old_unchanged"]
    assert copo["reference_unchanged"]
    assert (copo["gradient"] - grpo["gradient"]).norm().item() > 1e-5
    assert (copo["update"] - grpo["update"]).norm().item() > 1e-6
    # causal is nonzero only at reasoning column zero; answer/special columns receive no direct addition.
    assert torch.count_nonzero(causal[:, 1:]) == 0


def test_zero_causal_reward_exactly_reduces_to_grpo():
    initial = _Policy()
    first = _update(initial, torch.zeros(2, 3))
    second = _update(initial, torch.zeros(2, 3))
    assert torch.equal(first["gradient"], second["gradient"])
    assert torch.equal(first["update"], second["update"])


def test_causal_scale_has_controlled_nonzero_gradient_response():
    initial = _Policy()
    base = torch.tensor([[0.2, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    zero = _update(initial, torch.zeros_like(base))
    one = _update(initial, base)
    two = _update(initial, 2 * base)
    delta_one = one["gradient"] - zero["gradient"]
    delta_two = two["gradient"] - zero["gradient"]
    assert torch.allclose(delta_two, 2 * delta_one, atol=1e-6, rtol=1e-5)


def test_trainable_checkpoint_restores_output_optimizer_and_step(tmp_path):
    initial = _Policy()
    result = _update(initial, torch.tensor([[0.2, 0.0, 0.0], [-0.1, 0.0, 0.0]]))
    expected_output = result["model"].log_probs().detach().clone()
    path = tmp_path / "state.pt"
    save_trainable_checkpoint(
        path,
        result["model"],
        result["optimizer"],
        step=4,
        model_family="tiny",
        config={"lr": 0.05},
    )
    restored = _Policy()
    restored_optimizer = torch.optim.SGD([restored.weight], lr=0.05)
    step = load_trainable_checkpoint(
        path, restored, restored_optimizer, expected_model_family="tiny"
    )
    assert step == 4
    assert torch.equal(restored.log_probs().detach(), expected_output)
    assert torch.equal(restored.frozen.detach(), initial.frozen.detach())
    assert restored_optimizer.state_dict()["param_groups"] == result["optimizer"].state_dict()["param_groups"]


def test_checkpoint_rejects_wrong_model_family(tmp_path):
    model = _Policy()
    optimizer = torch.optim.SGD([model.weight], lr=0.05)
    path = tmp_path / "state.pt"
    save_trainable_checkpoint(path, model, optimizer, step=0, model_family="a", config={})
    with pytest.raises(ValueError, match="model family mismatch"):
        load_trainable_checkpoint(path, model, optimizer, expected_model_family="b")
