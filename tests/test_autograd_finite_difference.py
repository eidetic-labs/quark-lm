"""Permanent correctness proof for the scalar autodiff engine.

Finite-difference checks every backward rule in ``autograd.Scalar`` and the
full forward+backward loss surface. Pure scalar arithmetic, so it is fast and
deterministic. There was previously no numeric gradient check for the engine.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401  (inserts src/ onto sys.path)
from autograd import Scalar, zero_grad

EPS = 1e-6
OP_TOL = 1e-4
LOSS_TOL = 1e-3


def _expression(leaves: list[Scalar]) -> Scalar:
    """Exercises +, *, pow, log, exp, tanh, and truediv in one graph."""
    a, b = leaves
    return (
        (a * b + a.tanh()) * b.pow(2.0)
        + (a * 0.5 + b).exp()
        + (a * a + 1.0).log()
        + (a + 1.0) / (b * b + 2.0)
    )


class AutogradFiniteDifferenceTest(unittest.TestCase):
    def test_ops_match_finite_difference(self) -> None:
        point = [0.7, -0.4]
        leaves = [Scalar(v) for v in point]
        _expression(leaves).backward()
        analytic = [leaf.grad for leaf in leaves]

        for index, value in enumerate(point):
            def loss_at(scalar_value: float, index: int = index) -> float:
                perturbed = list(point)
                perturbed[index] = scalar_value
                return _expression([Scalar(v) for v in perturbed]).data

            numeric = (loss_at(value + EPS) - loss_at(value - EPS)) / (2 * EPS)
            self.assertAlmostEqual(
                analytic[index],
                numeric,
                delta=OP_TOL,
                msg=f"leaf {index}: analytic={analytic[index]} numeric={numeric}",
            )

    def test_loss_surface_gradient_matches_finite_difference(self) -> None:
        import random

        from support.char_model import char_model_fixture, context_and_target
        from transformer_probabilities import cross_entropy_scalars

        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=3)
        context, target = context_and_target(ids, config, tokenizer)
        params = model.parameters()

        zero_grad(params)
        cross_entropy_scalars(model._forward_scalars(context), target).backward()

        rng = random.Random(0)
        for index in rng.sample(range(len(params)), k=min(5, len(params))):
            param = params[index]
            analytic = param.grad
            original = param.data

            param.data = original + EPS
            plus = cross_entropy_scalars(model._forward_scalars(context), target).data
            param.data = original - EPS
            minus = cross_entropy_scalars(model._forward_scalars(context), target).data
            param.data = original

            numeric = (plus - minus) / (2 * EPS)
            self.assertAlmostEqual(
                analytic,
                numeric,
                delta=LOSS_TOL,
                msg=f"param {index}: analytic={analytic} numeric={numeric}",
            )


if __name__ == "__main__":
    unittest.main()
