import math
import unittest
from types import SimpleNamespace

from transformer_lm_attention import TransformerAttentionMixin


class AttentionHarness(TransformerAttentionMixin):
    def __init__(self) -> None:
        self.config = SimpleNamespace(embedding_dim=2, attention_heads=1)


class TransformerLMAttentionTest(unittest.TestCase):
    def test_causal_attention_floats_uses_only_past_positions(self) -> None:
        harness = AttentionHarness()
        q = [[1.0, 0.0], [1.0, 0.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[2.0, 3.0], [100.0, 200.0]]

        attended = harness._causal_attention_floats(q, k, v, 0)

        self.assertEqual(attended, [2.0, 3.0])

    def test_rotary_floats_leave_first_position_unchanged(self) -> None:
        harness = AttentionHarness()
        rows = [[1.0, 2.0], [3.0, 4.0]]

        rotated = harness._apply_rotary_floats(rows)

        self.assertEqual(rotated[0], rows[0])
        self.assertAlmostEqual(rotated[1][0], 3.0 * math.cos(1.0) - 4.0 * math.sin(1.0))
        self.assertAlmostEqual(rotated[1][1], 3.0 * math.sin(1.0) + 4.0 * math.cos(1.0))


if __name__ == "__main__":
    unittest.main()
