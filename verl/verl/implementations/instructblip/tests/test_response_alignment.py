"""Minimal unit tests for InstructBLIP response-only reward alignment.

No model is instantiated.  These tests exercise the pure tensor functions that
extract the response span from the full ``(image + instruction + response)``
sequence and align it to the offline causal-reward store, then verify that the
token-level causal reward enters the additive advantage (Eq. 9).

Run directly::

    python verl/verl/implementations/instructblip/tests/test_response_alignment.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import torch

# Load the torch-only COPO modules directly by file path so the test runs with
# just ``torch`` (no ray / transformers / model instantiation).  The alignment,
# causal-advantage and causal-reward modules have no intra-package imports, so
# they load independently without triggering ``verl/__init__``.
_IMPL = pathlib.Path(__file__).resolve().parents[1]  # .../implementations/instructblip


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register so @dataclass can resolve cls.__module__
    spec.loader.exec_module(module)
    return module


_alignment = _load_module("copo_alignment", _IMPL / "alignment.py")
_causal_advantage = _load_module("copo_causal_advantage", _IMPL / "causal" / "causal_advantage.py")
_causal_reward = _load_module("copo_causal_reward", _IMPL / "causal" / "causal_reward.py")
_rollout = _load_module("copo_instructblip_rollout", _IMPL / "rollout.py")

extract_response_only = _alignment.extract_response_only
append_response_tokens = _alignment.append_response_tokens
causal_advantage_from_group = _causal_advantage.causal_advantage_from_group
CausalRewardConfig = _causal_reward.CausalRewardConfig
build_causal_reward_tensor = _causal_reward.build_causal_reward_tensor
generated_continuation = _rollout._generated_continuation
generation_eos_id = _rollout._generation_eos_id

#: Store width the tests target (the offline reward store is [B, 2048]).
STORE_WIDTH = 2048


def _fake_full_sequence(batch: int, prefixes: list[int], resp_lens: list[int]):
    """Build fake full-sequence ``(B, T-1)`` log-probs and response mask.

    Row ``b`` has ``prefixes[b]`` non-response positions (image + instruction)
    followed by ``resp_lens[b]`` response tokens; the causal-shift already means
    index ``i`` describes full token ``i+1``.  ``gathered`` values are set to
    ``1..resp_len`` so extraction can be verified positionally.
    """
    seq_len = max(prefixes) + max(resp_lens)
    gathered = torch.zeros(batch, seq_len)
    response_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for b in range(batch):
        p, n = prefixes[b], resp_lens[b]
        response_mask[b, p:p + n] = True
        gathered[b, p:p + n] = torch.arange(1.0, n + 1.0)
    return gathered, response_mask


class TestExtractResponseOnly(unittest.TestCase):
    def test_exact_rollout_ids_define_labels(self):
        prompt_ids = torch.tensor([[101, 102, 103]])
        prompt_mask = torch.ones_like(prompt_ids)
        sampled = torch.tensor([201, 202])
        input_ids, attention_mask, labels = append_response_tokens(
            prompt_ids, prompt_mask, sampled, response_length=4
        )
        self.assertTrue(torch.equal(input_ids, torch.tensor([[101, 102, 103, 201, 202]])))
        self.assertTrue(torch.equal(attention_mask, torch.ones_like(input_ids)))
        self.assertTrue(torch.equal(labels, torch.tensor([[-100, -100, -100, 201, 202]])))

    def test_exact_rollout_ids_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeding response_length"):
            append_response_tokens(
                torch.tensor([[101]]), torch.tensor([[1]]), torch.tensor([201, 202]), response_length=1
            )

    def test_shape_is_response_only(self):
        gathered, response_mask = _fake_full_sequence(
            batch=4, prefixes=[3, 5, 2, 7], resp_lens=[11, 20, 5, 30],
        )
        out_logp, out_mask = extract_response_only(gathered, response_mask, STORE_WIDTH)
        self.assertEqual(out_logp.shape, (4, STORE_WIDTH))
        self.assertEqual(out_mask.shape, (4, STORE_WIDTH))

    def test_response_values_land_in_front_columns(self):
        gathered, response_mask = _fake_full_sequence(
            batch=3, prefixes=[2, 4, 1], resp_lens=[4, 6, 3],
        )
        out_logp, out_mask = extract_response_only(gathered, response_mask, STORE_WIDTH)
        for b, n in enumerate([4, 6, 3]):
            # The response fills the first ``n`` columns and nothing after.
            self.assertTrue(out_mask[b, :n].all(), f"row {b} response mask")
            self.assertFalse(out_mask[b, n:].any(), f"row {b} padding")
            self.assertTrue(torch.equal(out_logp[b, :n], torch.arange(1.0, n + 1.0)))
            self.assertTrue((out_logp[b, n:] == 0).all(), f"row {b} padding log-probs")

    def test_response_longer_than_store_is_rejected(self):
        gathered, response_mask = _fake_full_sequence(batch=1, prefixes=[0], resp_lens=[16])
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            extract_response_only(gathered, response_mask, 8)

    def test_empty_response_yields_empty_mask(self):
        gathered, response_mask = _fake_full_sequence(batch=1, prefixes=[4], resp_lens=[0])
        out_logp, out_mask = extract_response_only(gathered, response_mask, STORE_WIDTH)
        self.assertFalse(out_mask[0].any())
        self.assertTrue((out_logp[0] == 0).all())


class TestCausalRewardContract(unittest.TestCase):
    def test_mismatched_store_raises_clear_error(self):
        config = CausalRewardConfig()
        response_mask = torch.ones(2, STORE_WIDTH, dtype=torch.bool)
        bad_store = torch.zeros(2, 512)  # wrong response length
        with self.assertRaises(ValueError) as ctx:
            build_causal_reward_tensor(
                response_mask, config,
                sufficiency_reward=bad_store, necessity_reward=bad_store,
            )
        self.assertIn("response mask shape", str(ctx.exception))

    def test_reasoning_only_requires_explicit_verified_mask(self):
        config = CausalRewardConfig(reward_scope="reasoning_only")
        with self.assertRaisesRegex(ValueError, "explicit reasoning mask"):
            build_causal_reward_tensor(
                torch.ones(1, 2), config, causal_reward=torch.ones(1, 2)
            )

    def test_nonfinite_reward_is_rejected(self):
        config = CausalRewardConfig(reward_scope="reasoning_only")
        with self.assertRaisesRegex(ValueError, "NaN or inf"):
            build_causal_reward_tensor(
                torch.ones(1, 2), config,
                causal_reward=torch.tensor([[float("nan"), 0.0]]),
                reasoning_mask=torch.ones(1, 2),
            )

    def test_causal_reward_enters_additive_advantage(self):
        # Eq. 9: A_hat = A_orig + r_causal, masked to the response.
        B = 2
        response_mask = torch.zeros(B, STORE_WIDTH, dtype=torch.bool)
        response_mask[0, :10] = True
        response_mask[1, :5] = True
        group_adv = torch.tensor([0.5, -0.5])
        causal_reward = torch.zeros(B, STORE_WIDTH)
        causal_reward[0, :10] = 0.1
        causal_reward[1, :5] = 0.2

        advantage = causal_advantage_from_group(group_adv, response_mask, causal_reward)

        self.assertEqual(advantage.shape, (B, STORE_WIDTH))
        self.assertTrue(torch.allclose(advantage[0, :10], torch.full((10,), 0.6)))  # 0.5 + 0.1
        self.assertTrue((advantage[0, 10:] == 0).all())
        self.assertTrue(torch.allclose(advantage[1, :5], torch.full((5,), -0.3)))  # -0.5 + 0.2
        self.assertTrue((advantage[1, 5:] == 0).all())


class TestGenerationOutputLayout(unittest.TestCase):
    def test_decoder_only_prompt_is_removed(self):
        output = torch.tensor([10, 11, 20, 21])
        self.assertTrue(torch.equal(generated_continuation(output, 2, False), torch.tensor([20, 21])))

    def test_encoder_decoder_output_is_not_sliced(self):
        output = torch.tensor([0, 20, 21])
        self.assertTrue(torch.equal(generated_continuation(output, 7, True), output))

    def test_generation_config_eos_takes_precedence_over_tokenizer_metadata(self):
        model = type("Model", (), {"generation_config": type("Config", (), {"eos_token_id": 1})()})()
        tokenizer = type("Tokenizer", (), {"eos_token_id": 2})()
        self.assertEqual(generation_eos_id(model, tokenizer), 1)


if __name__ == "__main__":
    unittest.main()
