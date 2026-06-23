"""LR-schedule variants for scheduled_learning_rate (default-off parity + cosine/wsd).

The default schedule="linear" path must reproduce today's warmup-then-linear-decay
values BYTE-FOR-BYTE (the parity-validated trainers depend on it). The cosine tail
must be monotone non-increasing post-warmup and land on min_learning_rate at the end.
"""

from __future__ import annotations

import math
import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from transformer_optimizer import scheduled_learning_rate


def _legacy_linear(base, step, *, warmup_steps, decay_steps, min_learning_rate):
    """The pre-change implementation, inlined as the parity oracle."""

    learning_rate = base
    if warmup_steps > 0:
        learning_rate *= min(1.0, step / warmup_steps)
    if decay_steps > 0 and step > warmup_steps:
        decay_step = min(step - warmup_steps, decay_steps)
        decay_fraction = decay_step / decay_steps
        learning_rate = learning_rate - (learning_rate - min_learning_rate) * decay_fraction
    return max(learning_rate, min_learning_rate)


class LinearScheduleParityTest(unittest.TestCase):
    def test_default_reproduces_legacy_linear_exactly(self) -> None:
        base, warmup, decay, min_lr = 0.001, 80, 800, 0.0001
        for step in range(0, 1000):
            expected = _legacy_linear(
                base, step, warmup_steps=warmup, decay_steps=decay, min_learning_rate=min_lr
            )
            got = scheduled_learning_rate(
                base, step, warmup_steps=warmup, decay_steps=decay, min_learning_rate=min_lr
            )
            self.assertEqual(got, expected, f"step {step}")

    def test_explicit_linear_matches_default(self) -> None:
        base, warmup, decay, min_lr = 0.005, 50, 500, 0.0
        for step in (0, 25, 50, 100, 300, 550):
            default = scheduled_learning_rate(
                base, step, warmup_steps=warmup, decay_steps=decay, min_learning_rate=min_lr
            )
            explicit = scheduled_learning_rate(
                base, step, warmup_steps=warmup, decay_steps=decay,
                min_learning_rate=min_lr, schedule="linear",
            )
            self.assertEqual(default, explicit)


class CosineScheduleTest(unittest.TestCase):
    def test_cosine_monotone_decreasing_post_warmup_and_ends_at_min(self) -> None:
        base, warmup, decay, min_lr = 0.01, 100, 1000, 0.0005
        # Warmup itself is shared with linear: at warmup end, lr == base (peak).
        peak = scheduled_learning_rate(
            base, warmup, warmup_steps=warmup, decay_steps=decay,
            min_learning_rate=min_lr, schedule="cosine",
        )
        self.assertAlmostEqual(peak, base)
        previous = peak
        for step in range(warmup + 1, warmup + decay + 1):
            value = scheduled_learning_rate(
                base, step, warmup_steps=warmup, decay_steps=decay,
                min_learning_rate=min_lr, schedule="cosine",
            )
            self.assertLessEqual(value, previous + 1e-12, f"non-monotone at {step}")
            previous = value
        end = scheduled_learning_rate(
            base, warmup + decay, warmup_steps=warmup, decay_steps=decay,
            min_learning_rate=min_lr, schedule="cosine",
        )
        self.assertAlmostEqual(end, min_lr)

    def test_cosine_midpoint_is_half_amplitude(self) -> None:
        base, warmup, decay, min_lr = 0.01, 0, 1000, 0.0
        mid = scheduled_learning_rate(
            base, 500, warmup_steps=warmup, decay_steps=decay,
            min_learning_rate=min_lr, schedule="cosine",
        )
        expected = min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * 0.5))
        self.assertAlmostEqual(mid, expected)


class WsdScheduleTest(unittest.TestCase):
    def test_wsd_holds_peak_then_cosine_decays_to_min(self) -> None:
        base, warmup, decay, min_lr = 0.01, 50, 1000, 0.0
        # Early in the tail wsd holds the peak (stable phase) -> still == base.
        held = scheduled_learning_rate(
            base, warmup + 10, warmup_steps=warmup, decay_steps=decay,
            min_learning_rate=min_lr, schedule="wsd",
        )
        self.assertAlmostEqual(held, base)
        # By the end it has decayed to min via the cosine tail.
        end = scheduled_learning_rate(
            base, warmup + decay, warmup_steps=warmup, decay_steps=decay,
            min_learning_rate=min_lr, schedule="wsd",
        )
        self.assertAlmostEqual(end, min_lr)


if __name__ == "__main__":
    unittest.main()
