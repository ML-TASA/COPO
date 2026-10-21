from __future__ import annotations

import pytest
import torch

from verl.utils.causal_reward.adapter import compute_fixed_copo_advantages, prepare_fixed_adapter_batch
from verl.utils.causal_reward.tests.synthetic_v3 import make_synthetic_v3_store
from verl.utils.causal_reward.trajectory import FixedTrajectoryStore


def _selected():
    store = make_synthetic_v3_store()
    return FixedTrajectoryStore(store).select_groups(
        ["sample-0"], ["group-0"], ["image-0"], ["d" * 64]
    )


def test_adapter_bridge_preserves_exact_action_and_reasoning_only_advantage():
    selected = _selected()
    ids = selected._tensor("response_token_ids")
    mask = selected._tensor("response_mask")
    batch = prepare_fixed_adapter_batch(
        selected,
        prompt_token_ids=[11, 12],
        response_token_ids=ids,
        response_mask=mask,
        device="cpu",
    )
    advantage, original = compute_fixed_copo_advantages(batch)
    assert torch.equal(batch.response_token_ids, ids)
    assert torch.equal(batch.response_mask, mask.bool())
    assert torch.allclose(advantage[batch.answer_mask], original[batch.answer_mask])
    assert torch.allclose(advantage[batch.special_token_mask], original[batch.special_token_mask])
    assert torch.allclose(
        advantage[batch.reasoning_mask],
        original[batch.reasoning_mask] + batch.causal_rewards[batch.reasoning_mask],
    )
    assert torch.count_nonzero(advantage[~batch.response_mask]) == 0


@pytest.mark.parametrize("field", ["tokens", "mask", "prompt"])
def test_adapter_bridge_fails_closed_on_identity_changes(field):
    selected = _selected()
    ids = selected._tensor("response_token_ids").clone()
    mask = selected._tensor("response_mask").clone()
    prompt = [11, 12]
    if field == "tokens":
        ids[0, 0] += 1
    elif field == "mask":
        mask[0, 4] = 0
    else:
        prompt[0] += 1
    with pytest.raises(ValueError, match="fixed adapter"):
        prepare_fixed_adapter_batch(
            selected,
            prompt_token_ids=prompt,
            response_token_ids=ids,
            response_mask=mask,
            device="cpu",
        )
