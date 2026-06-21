"""Parity: torch two-group AdamW with exclusion matches the scalar masked optimizer.

At weight_decay>0 the scalar engine (per-element no_decay mask) and the torch
two-group AdamW must exclude the SAME tensors and stay parity-equal. A non-zero
warmup makes the LR schedule cross multiple values, which also guards the
all-param-groups LR write -- a param_groups[0]-only write would leave the no-decay
group on a stale LR and diverge. Skip-safe when torch is not installed.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from support.char_model import char_model_fixture, context_and_target
from support.core import OptimizationConfig
from transformer_no_decay_mask import build_no_decay_mask
from transformer_training_parameter_manifest import build_training_parameter_manifest
from transformer_training_parity_fixture import build_scalar_training_parity_fixture
from transformer_torch_training_loop import eval_torch_loss, train_torch_lm

RUNTIME = {"dtype": "float64", "device": "cpu"}


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class WeightDecayExclusionParityTest(unittest.TestCase):
    def _run(self, *, accum: int) -> tuple[float, float]:
        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
        context, target = context_and_target(ids, config, tokenizer)
        fixture = build_scalar_training_parity_fixture(
            fixture_id="wd-excl", model=model, tokenizer=tokenizer, context=context, target=target,
            optimizer_config=OptimizationConfig(
                optimizer="adamw", weight_decay=0.05, gradient_accumulation_steps=accum, warmup_steps=3,
            ),
            learning_rate=0.1, steps=8, corpus_hash="x",
        )
        state, _losses = train_torch_lm(
            fixture=fixture, examples=[(context, target)], steps=8,
            learning_rate=0.1, torch=torch, runtime=RUNTIME,
        )
        final = eval_torch_loss(
            fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=RUNTIME,
        )
        return final, fixture["training_case"]["final_loss"]

    def test_mask_is_a_real_partition(self) -> None:
        # Negative control: the no_decay mask contains BOTH decayed and excluded
        # elements, so a regression that masks everything/nothing fails here rather
        # than passing on mutual scalar/torch agreement.
        _tokenizer, _ids, config, model = char_model_fixture("abc abc\n", seed=53)
        manifest = build_training_parameter_manifest(
            weights=model.to_dict()["weights"], model_config=asdict(config)
        )
        mask = build_no_decay_mask(manifest)
        self.assertIn(True, mask)
        self.assertIn(False, mask)

    def test_exclusion_parity_no_accumulation(self) -> None:
        final, scalar_final = self._run(accum=1)
        self.assertAlmostEqual(final, scalar_final, delta=1e-6)

    def test_exclusion_parity_with_accumulation(self) -> None:
        final, scalar_final = self._run(accum=2)
        self.assertAlmostEqual(final, scalar_final, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
