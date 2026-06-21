"""Behavioral test: weight-decay exclusion actually skips decay for masked params.

Isolates the decay term from the Adam term -- with zero gradients the Adam update
is exactly 0, so only weight decay can move a parameter -- proving the exclusion
MECHANISM independent of any torch==scalar agreement (a parity test alone can't
catch both backends wrongly decaying or wrongly skipping the same way).
"""

from __future__ import annotations

import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from autograd import Scalar
from support.core import OptimizationConfig
from transformer_optimizer import ScalarOptimizer

LR = 0.5


def _apply_once(mask: list[bool], weight_decay: float) -> tuple[float, float]:
    config = OptimizationConfig(
        optimizer="adamw",
        weight_decay=weight_decay,
        gradient_clip=0.0,
        gradient_accumulation_steps=1,
    )
    optimizer = ScalarOptimizer(config, no_decay_mask=mask)
    params = [Scalar(1.0), Scalar(1.0)]
    for parameter in params:
        parameter.grad = 0.0  # zero grad -> Adam term is exactly 0; only decay moves
    optimizer.apply(params, LR)
    return params[0].data, params[1].data


class WeightDecayExclusionBehaviorTest(unittest.TestCase):
    def test_no_decay_param_is_not_decayed(self) -> None:
        # mask = [decay, no_decay]; with zero grads only decay can move a param.
        decayed, excluded = _apply_once([False, True], 0.1)
        self.assertAlmostEqual(decayed, 1.0 - LR * 0.1)  # decay fired
        self.assertEqual(excluded, 1.0)  # decay skipped -> EXACTLY unchanged

    def test_zero_weight_decay_moves_nothing(self) -> None:
        first, second = _apply_once([False, True], 0.0)
        self.assertEqual(first, 1.0)
        self.assertEqual(second, 1.0)

    def test_empty_mask_decays_everything(self) -> None:
        # Empty mask == uniform (pre-exclusion) behavior: both shrink identically.
        first, second = _apply_once([], 0.1)
        self.assertAlmostEqual(first, 1.0 - LR * 0.1)
        self.assertAlmostEqual(second, 1.0 - LR * 0.1)


if __name__ == "__main__":
    unittest.main()
