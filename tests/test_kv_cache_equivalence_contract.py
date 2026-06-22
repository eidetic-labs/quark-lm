"""Phase-0 contract for the unbounded RoPE + append-valid KV-cache architecture.

These tests encode, BEFORE the architecture exists, the invariants the phased build
must preserve -- so a half-built flag can never silently no-op and still look green
(the exact regression class the recovery work exists to prevent):

  1. The KV-cache decode path is a pure no-op-equivalent of the uncached path
     (byte-identical generated tokens). This is the anchor the real append-valid
     cache (Phase 3) must keep bit-exact; today the cache flag is a telemetry stub,
     so the invariant holds trivially and becomes load-bearing when the cache lands.
  2. The use_prompt_position_projection summand is present and NONZERO under the
     right-anchored geometry, so a future make-context-free / re-keyed recompute that
     drops or mis-indexes it is detectably wrong (adversarial verdicts 2 and 3).
  3. Every not-yet-implemented positional/cache flag FAILS CLOSED at model
     construction (raises) rather than silently doing nothing.
  4. An evicting window combined with the position-projection readout is REFUSED, not
     silently zeroed (verdict 1: an evicting K/V cache cannot reconstruct the
     residual rows the projection reads -- K is non-invertible).
  5. make_context_positioned is a pure superset of make_context: identical window
     content, plus each slot's ABSOLUTE sequence index (pads get a negative sentinel).
"""

from __future__ import annotations

import random
import unittest

import support  # noqa: F401  (puts src/ on sys.path)
from neural_char_ops import POSITION_PAD_SENTINEL, make_context, make_context_positioned
from tokenizer import CharTokenizer
from transformer_kv_cache_contract import kv_cache_contract_violation
from transformer_model import GenerationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM

CORPUS = "mia ball box red cup noah shelf\n"
UNBUILT_FLAGS = (
    "use_all_positions_causal",
    "kv_cache_stores_summary_state",
)


def _model(tokenizer: CharTokenizer, **overrides) -> TinyTransformerLM:
    defaults = dict(
        vocab_size=tokenizer.vocab_size, context_size=6, embedding_dim=4,
        feedforward_dim=8, attention_heads=2, seed=7,
    )
    defaults.update(overrides)
    return TinyTransformerLM.init_random(TransformerConfig(**defaults))


def _randomize_position_projection(model: TinyTransformerLM, seed: int) -> None:
    """The projection weights init to zero Scalars; set nonzero ``.data`` to exercise."""

    rand = random.Random(seed)

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif hasattr(value, "data"):
            value.data = rand.uniform(-0.3, 0.3)

    walk(model.prompt_position_projection_w)
    walk(model.prompt_position_projection_b)


class KvCacheContractTest(unittest.TestCase):
    def test_kv_cache_stub_matches_uncached(self) -> None:
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer)
        cached = model.generate_with_trace(
            tokenizer, "mia", 5, GenerationConfig(use_kv_cache=True)
        )
        uncached = model.generate_with_trace(
            tokenizer, "mia", 5, GenerationConfig(use_kv_cache=False)
        )
        self.assertEqual(cached["text"], uncached["text"])
        self.assertEqual(
            [step["token_id"] for step in cached["trace"]],
            [step["token_id"] for step in uncached["trace"]],
        )

    def test_position_projection_summand_present_and_nonzero(self) -> None:
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_prompt_position_projection=True)
        _randomize_position_projection(model, seed=21)
        context = make_context(tokenizer.encode("mia ball"), 6, tokenizer.pad_id)
        with_projection = model.predict(context)
        model.config.use_prompt_position_projection = False
        without_projection = model.predict(context)
        delta = max(abs(a - b) for a, b in zip(with_projection, without_projection))
        self.assertGreater(delta, 1e-9)

    def test_unbuilt_flags_fail_closed(self) -> None:
        tokenizer = CharTokenizer.train(CORPUS)
        for flag in UNBUILT_FLAGS:
            with self.assertRaises(ValueError, msg=f"{flag} should fail closed"):
                _model(tokenizer, **{flag: True})

    def test_absolute_rope_constructs(self) -> None:
        # Phase 1 is built: use_absolute_rope no longer fails closed at construction.
        # Phase 2 (R-B): use_absolute_rope drops the learned pos-embed addend, so RoPE is
        # the sole positional source -- it REQUIRES use_rotary_positions=True, else the
        # config is position-blind and refused.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(
            tokenizer, use_absolute_rope=True, use_rotary_positions=True
        )  # must not raise
        self.assertTrue(model.config.use_absolute_rope)
        self.assertIsNone(
            kv_cache_contract_violation(
                {"use_absolute_rope": True, "use_rotary_positions": True}
            )
        )
        # Position-blind config (absolute RoPE, no RoPE) is refused structurally.
        self.assertIsNotNone(kv_cache_contract_violation({"use_absolute_rope": True}))

    def test_sliding_window_fails_closed(self) -> None:
        tokenizer = CharTokenizer.train(CORPUS)
        with self.assertRaises(ValueError):
            _model(tokenizer, sliding_window_size=4)

    def test_default_config_does_not_fail_closed(self) -> None:
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer)  # must not raise
        self.assertIsNone(kv_cache_contract_violation({}))
        self.assertEqual(model.config.context_size, 6)

    def test_eviction_with_projection_refused(self) -> None:
        reason = kv_cache_contract_violation(
            {"use_prompt_position_projection": True, "sliding_window_size": 4}
        )
        self.assertIsNotNone(reason)

    def test_make_context_positioned_short_sequence(self) -> None:
        ids = [5, 6, 7]
        context, positions = make_context_positioned(ids, 6, 0)
        self.assertEqual(context, make_context(ids, 6, 0))
        self.assertEqual(context, [0, 0, 0, 5, 6, 7])
        self.assertEqual(positions, [POSITION_PAD_SENTINEL] * 3 + [0, 1, 2])

    def test_make_context_positioned_long_sequence_absolute_index(self) -> None:
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        context, positions = make_context_positioned(ids, 6, 0)
        self.assertEqual(context, make_context(ids, 6, 0))
        self.assertEqual(context, [3, 4, 5, 6, 7, 8])
        # Absolute indices of the right-anchored window -- they grow with the stream
        # and never re-index, which is what makes an append cache valid (Phase 1+).
        self.assertEqual(positions, [2, 3, 4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
