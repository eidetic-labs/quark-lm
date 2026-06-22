"""SDPA attention path == hand-rolled attention (CPU real-torch, within 1e-6).

The opt-in fused attention (runtime['use_sdpa']) computes the same
softmax((q.k)*scale).v as the hand-rolled head, so on identical weights it must
agree within the float64 parity band. Default-off, so the fake-torch double and
existing parity are unaffected; this test exercises the SDPA path on real torch.
"""

from __future__ import annotations

import unittest
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from support.char_model import char_model_fixture, context_and_target
from support.core import OptimizationConfig
from transformer_training_parity_fixture import build_scalar_training_parity_fixture
from transformer_torch_training_loop import eval_torch_loss
from transformer_torch_training_state import build_torch_training_state

CPU = {"dtype": "float64", "device": "cpu"}
SDPA = {"dtype": "float64", "device": "cpu", "use_sdpa": True}


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class SdpaAttentionParityTest(unittest.TestCase):
    def test_sdpa_matches_hand_rolled_attention(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
        context, target = context_and_target(ids, config, tokenizer)
        fixture = build_scalar_training_parity_fixture(
            fixture_id="sdpa", model=model, tokenizer=tokenizer, context=context, target=target,
            optimizer_config=OptimizationConfig(
                optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
            ),
            learning_rate=0.05, steps=1, corpus_hash="x",
        )
        # Same weights, two attention implementations -> losses must agree.
        state = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU)
        hand_rolled = eval_torch_loss(
            fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=CPU
        )
        sdpa = eval_torch_loss(
            fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=SDPA
        )
        self.assertAlmostEqual(hand_rolled, sdpa, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
