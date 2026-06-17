import unittest
from types import SimpleNamespace

from transformer_lm_feedforward import TransformerFeedForwardMixin


class FeedForwardHarness(TransformerFeedForwardMixin):
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            embedding_dim=2,
            layer_norm_epsilon=1e-5,
            use_gated_mlp=False,
            use_layer_norm=False,
            use_pre_layer_norm=False,
            use_rms_norm=False,
        )


def zero_block() -> dict[str, list[list[float]] | list[float]]:
    return {
        "ln1_gain": [1.0, 1.0],
        "ln1_bias": [0.0, 0.0],
        "ln2_gain": [1.0, 1.0],
        "ln2_bias": [0.0, 0.0],
        "w1": [[0.0, 0.0], [0.0, 0.0]],
        "b1": [0.0, 0.0],
        "w_gate": [[0.0, 0.0], [0.0, 0.0]],
        "b_gate": [0.0, 0.0],
        "w2": [[0.0, 0.0], [0.0, 0.0]],
        "b2": [0.0, 0.0],
    }


class TransformerLMFeedForwardTest(unittest.TestCase):
    def test_feed_forward_floats_preserves_hidden_when_projection_is_zero(self) -> None:
        harness = FeedForwardHarness()

        output = harness._feed_forward_floats([1.5, -2.0], zero_block())

        self.assertEqual(output, [1.5, -2.0])

    def test_attention_input_floats_skips_norm_when_pre_norm_is_disabled(self) -> None:
        harness = FeedForwardHarness()
        rows = [[1.0, 2.0], [3.0, 4.0]]

        output = harness._attention_input_floats(rows, zero_block())

        self.assertIs(output, rows)


if __name__ == "__main__":
    unittest.main()
