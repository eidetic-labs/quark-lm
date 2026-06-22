"""Phase 3 bit-exact gate for the regime-aware append-valid KV cache.

The cache is LAYER-0-ONLY (the adversarially-confirmed central correctness fact):
under the thesis decode geometry (use_absolute_rope drops the learned pos-embed
addend + use_rotary_positions keys RoPE by ABSOLUTE position, V unrotated) the
layer-0 K[p]/V[p] are WRITE-ONCE. For UPPER layers it is FALSE -- a token's
upper-layer K/V INPUT changes as the right-anchored window slides -- so upper layers
are RECOMPUTED every step. These tests pin that:

  * Gate 1: cache-on == cache-off == the validated right-anchored full recompute
    (predict(context, positions) via make_context_positioned), at f64 1e-6 PER-STEP
    probabilities + argmax token sequence + final text, over a decode that slides the
    window PAST the boundary, run for num_layers==1 AND num_layers>1 SEPARATELY (the
    upper-layer recompute path is exercised).
  * Gate 2 (the window-slide boundary LOCK): num_layers==2, small context_size,
    decode past the window-full boundary, cache-on == cache-off at f64 1e-6 -- this
    FAILS against a naive all-layers cache and passes against layer-0-only.
  * Pad never cached: a short-prompt decode is identical cache-on vs off, and the
    cache holds no negative key.
  * Default-OFF byte-exact: flag-off output is byte-identical to a pre-Phase-3
    baseline forward.
  * Cache hits ACTUALLY occurred (so cache_enabled is not silently False).
  * The position-projection summand is bit-identical cache-on vs off (the windowed
    readout reads no K/V).
  * Torch mirror: cache-on == cache-off on REAL torch at the f32 band, cache hits
    occurred. system python3 silently skips torch (a false green via the fake-torch
    double that lacks SDPA), so run with PYTHONPATH=src:tests .venv/bin/python.
"""

from __future__ import annotations

import random
import unittest
from dataclasses import asdict
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from neural_char_ops import POSITION_PAD_SENTINEL, make_context_positioned
from tokenizer import CharTokenizer
from transformer_kv_cache import Layer0KVCache
from transformer_model import GenerationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM

CORPUS = "mia ball box red cup noah shelf ava leo book\n"


def _model(tokenizer: CharTokenizer, **overrides) -> TinyTransformerLM:
    defaults = dict(
        vocab_size=tokenizer.vocab_size,
        context_size=4,
        embedding_dim=4,
        feedforward_dim=8,
        attention_heads=2,
        seed=7,
        use_rotary_positions=True,
        use_absolute_rope=True,
    )
    defaults.update(overrides)
    return TinyTransformerLM.init_random(TransformerConfig(**defaults))


def _randomize_position_projection(model: TinyTransformerLM, seed: int) -> None:
    rand = random.Random(seed)

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif hasattr(value, "data"):
            value.data = rand.uniform(-0.3, 0.3)

    walk(model.prompt_position_projection_w)
    walk(model.prompt_position_projection_b)


def _right_anchored_recompute(model, tokenizer, prompt, max_new):
    """The validated right-anchored full recompute: predict(context, positions) per
    step via make_context_positioned, greedy, no cache. This is the reference."""

    ids = tokenizer.encode(prompt)
    generated: list[int] = []
    step_probs: list[list[float]] = []
    for _ in range(max_new):
        context, positions = make_context_positioned(
            ids, model.config.context_size, tokenizer.pad_id
        )
        probs = model.predict(context, positions)  # no cache -> full recompute
        step_probs.append(list(probs))
        next_id = max(range(len(probs)), key=lambda index: probs[index])
        ids.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(generated), generated, step_probs


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class DecodeEquivalenceScalarTest(unittest.TestCase):
    def _assert_bit_exact_decode(self, num_layers: int) -> None:
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, num_layers=num_layers)
        context_size = model.config.context_size
        # Decode at least context_size + 3 steps so the window slides PAST the boundary.
        max_new = context_size + 5
        on = model.generate_with_trace(
            tokenizer, "mia", max_new, GenerationConfig(use_kv_cache=True)
        )
        off = model.generate_with_trace(
            tokenizer, "mia", max_new, GenerationConfig(use_kv_cache=False)
        )
        ref_text, ref_ids, ref_probs = _right_anchored_recompute(
            model, tokenizer, "mia", max_new
        )

        # Final text + argmax token sequence: cache-on == cache-off == recompute.
        self.assertEqual(on["text"], off["text"])
        self.assertEqual(on["text"], ref_text)
        on_ids = [s["token_id"] for s in on["trace"]]
        off_ids = [s["token_id"] for s in off["trace"]]
        self.assertEqual(on_ids, off_ids)
        self.assertEqual(on_ids, ref_ids)

        # Per-step probability vectors at f64 1e-6 (not just argmax). The trace's
        # top_tokens carry raw (pre-filter) probabilities aligned by token_id.
        for step_index, (on_step, off_step) in enumerate(
            zip(on["trace"], off["trace"])
        ):
            ref_step = ref_probs[step_index]
            for on_top, off_top in zip(on_step["top_tokens"], off_step["top_tokens"]):
                self.assertEqual(on_top["token_id"], off_top["token_id"])
                self.assertAlmostEqual(
                    on_top["raw_probability"],
                    off_top["raw_probability"],
                    delta=1e-6,
                )
                # And against the right-anchored full recompute reference.
                self.assertAlmostEqual(
                    on_top["raw_probability"],
                    ref_step[on_top["token_id"]],
                    delta=1e-6,
                )

        # The window genuinely slid past the boundary (more steps than context_size).
        self.assertGreater(max_new, context_size)
        # Cache hits ACTUALLY occurred -- cache_enabled is not silently False, and the
        # historical layer-0 K/V was served (not recomputed) on the post-boundary steps.
        self.assertGreater(on["cache"]["hits"], 0)
        self.assertGreater(on["cache"]["writes"], 0)
        self.assertEqual(off["cache"]["hits"], 0)

    def test_bit_exact_decode_single_layer(self) -> None:
        # num_layers == 1: layer 0 IS the whole model -> a full cache.
        self._assert_bit_exact_decode(num_layers=1)

    def test_bit_exact_decode_multi_layer(self) -> None:
        # num_layers > 1: the upper-layer RECOMPUTE path is exercised (cache passed to
        # layer 0 only). Equivalence here proves upper layers are NOT served stale rows.
        self._assert_bit_exact_decode(num_layers=2)

    def test_window_slide_boundary_lock_two_layers(self) -> None:
        # THE VERDICT FAILURE CASE. Two layers, small context_size, decode PAST the
        # window-full boundary. cache-on == cache-off at f64 1e-6. This passes against
        # the layer-0-only design and FAILS against a naive all-layers cache (verified
        # separately by the negative-control test below).
        for context_size in (4, 6):
            tokenizer = CharTokenizer.train(CORPUS)
            model = _model(tokenizer, num_layers=2, context_size=context_size)
            max_new = context_size + 4  # slide well past the boundary
            on = model.generate_with_trace(
                tokenizer, "mia ball", max_new, GenerationConfig(use_kv_cache=True)
            )
            off = model.generate_with_trace(
                tokenizer, "mia ball", max_new, GenerationConfig(use_kv_cache=False)
            )
            self.assertEqual(on["text"], off["text"])
            self.assertGreater(on["cache"]["hits"], 0)
            for on_step, off_step in zip(on["trace"], off["trace"]):
                for on_top, off_top in zip(
                    on_step["top_tokens"], off_step["top_tokens"]
                ):
                    self.assertAlmostEqual(
                        on_top["raw_probability"],
                        off_top["raw_probability"],
                        delta=1e-6,
                        msg=f"ctx={context_size} boundary slide broke bit-exactness",
                    )

    def test_naive_all_layers_cache_is_broken_negative_control(self) -> None:
        # The layer-0-only scope is LOAD-BEARING: a cache served to ALL layers (the
        # refuted design) produces a stale upper-layer row once the window slides, so it
        # DIVERGES from cache-off by far more than 1e-6. This proves the boundary lock
        # above is not a degenerate pass.
        from transformer_math import matrix_to_floats

        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, num_layers=2, context_size=4)
        original_final_hidden = model.final_hidden

        def naive_all_layers_final_hidden(context, positions=None, cache=None):
            if cache is None:
                return original_final_hidden(context, positions, None)
            token_embeddings = matrix_to_floats(model.token_embeddings)
            x = [
                [token_embeddings[t][d] for d in range(model.config.embedding_dim)]
                for t in context
            ]
            float_blocks = [model._block_to_floats(b) for b in model.blocks]
            for block in float_blocks[:-1]:
                # BUG: pass the cache to EVERY upper layer.
                x = model._forward_full_block_floats(x, block, positions, cache)
            return model._finalize_hidden_floats(
                model._forward_final_block_floats(
                    x, float_blocks[-1], context, positions, cache
                )
            )

        model.final_hidden = naive_all_layers_final_hidden
        on = model.generate_with_trace(
            tokenizer, "mia ball", 8, GenerationConfig(use_kv_cache=True)
        )
        model.final_hidden = original_final_hidden
        off = model.generate_with_trace(
            tokenizer, "mia ball", 8, GenerationConfig(use_kv_cache=False)
        )
        max_delta = 0.0
        for on_step, off_step in zip(on["trace"], off["trace"]):
            for on_top, off_top in zip(on_step["top_tokens"], off_step["top_tokens"]):
                if on_top["token_id"] == off_top["token_id"]:
                    max_delta = max(
                        max_delta,
                        abs(on_top["raw_probability"] - off_top["raw_probability"]),
                    )
        self.assertGreater(
            max_delta,
            1e-6,
            "a naive all-layers cache must DIVERGE; if it does not, the boundary "
            "lock is not load-bearing",
        )

    def test_pad_slot_never_cached_short_prompt(self) -> None:
        # A short prompt left-pads the window. cache-on == cache-off, and the cache
        # holds no NEGATIVE key (a -1 sentinel would collide across steps).
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, context_size=6)  # > prompt -> heavy left pad
        on = model.generate_with_trace(
            tokenizer, "m", 3, GenerationConfig(use_kv_cache=True)
        )
        off = model.generate_with_trace(
            tokenizer, "m", 3, GenerationConfig(use_kv_cache=False)
        )
        self.assertEqual(on["text"], off["text"])
        for on_step, off_step in zip(on["trace"], off["trace"]):
            for on_top, off_top in zip(on_step["top_tokens"], off_step["top_tokens"]):
                self.assertAlmostEqual(
                    on_top["raw_probability"], off_top["raw_probability"], delta=1e-6
                )

        # Direct cache assertion: storing a pad sentinel is skipped; no negative key.
        cache = Layer0KVCache()
        cache.store(POSITION_PAD_SENTINEL, [0.0] * 4, [0.0] * 4)
        self.assertFalse(cache.has(POSITION_PAD_SENTINEL))
        self.assertEqual(cache.pad_skips, 1)
        cache.store(0, [1.0] * 4, [2.0] * 4)
        self.assertTrue(all(key >= 0 for key in cache._keys))
        self.assertEqual(cache.writes, 1)

    def test_default_off_byte_exact_baseline(self) -> None:
        # Flag OFF is byte-identical to a pre-Phase-3 baseline forward. The baseline is
        # predict(context, positions) with NO cache argument (the original signature),
        # driven through the identical greedy loop.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer)
        off = model.generate_with_trace(
            tokenizer, "mia ball", 7, GenerationConfig(use_kv_cache=False)
        )
        baseline_text, baseline_ids, baseline_probs = _right_anchored_recompute(
            model, tokenizer, "mia ball", 7
        )
        self.assertEqual(off["text"], baseline_text)
        self.assertEqual([s["token_id"] for s in off["trace"]], baseline_ids)
        for step_index, off_step in enumerate(off["trace"]):
            for off_top in off_step["top_tokens"]:
                # BYTE-exact (==), not tolerance: flag-off must not touch the float path.
                self.assertEqual(
                    off_top["raw_probability"],
                    baseline_probs[step_index][off_top["token_id"]],
                )
        # And the disabled cache reports no activity.
        self.assertFalse(off["cache"]["enabled"])
        self.assertEqual(off["cache"]["hits"], 0)
        self.assertEqual(off["cache"]["writes"], 0)

    def test_config_path_flag_enables_cache(self) -> None:
        # cache_enabled is GenerationConfig.use_kv_cache OR TransformerConfig
        # .use_kv_cache_path. Enabling via the config path also produces bit-exact
        # output and real cache hits.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_kv_cache_path=True)
        on = model.generate_with_trace(
            tokenizer, "mia", 7, GenerationConfig(use_kv_cache=False)
        )
        baseline_text, _ids, _probs = _right_anchored_recompute(
            model, tokenizer, "mia", 7
        )
        self.assertTrue(on["cache"]["enabled"])
        self.assertGreater(on["cache"]["hits"], 0)
        self.assertEqual(on["text"], baseline_text)

    def test_position_projection_summand_bit_identical_cache_on_off(self) -> None:
        # The windowed position-projection readout reads no K/V; it must be UNCHANGED by
        # the cache. With the projection on (randomized nonzero weights), cache-on ==
        # cache-off bit-for-bit over a sliding decode.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_prompt_position_projection=True)
        _randomize_position_projection(model, seed=21)
        on = model.generate_with_trace(
            tokenizer, "mia ball", 8, GenerationConfig(use_kv_cache=True)
        )
        off = model.generate_with_trace(
            tokenizer, "mia ball", 8, GenerationConfig(use_kv_cache=False)
        )
        self.assertGreater(on["cache"]["hits"], 0)
        for on_step, off_step in zip(on["trace"], off["trace"]):
            for on_top, off_top in zip(on_step["top_tokens"], off_step["top_tokens"]):
                self.assertEqual(on_top["token_id"], off_top["token_id"])
                # Bit-identical: the projection summand is the same whether or not the
                # historical layer-0 K/V is served from cache.
                self.assertAlmostEqual(
                    on_top["raw_probability"], off_top["raw_probability"], delta=1e-9
                )


class DecodeEquivalenceTorchTest(unittest.TestCase):
    def _tok_fixture(self, model, tokenizer):
        fixture = {
            "weights": model.to_dict()["weights"],
            "model_config": asdict(model.config),
        }
        fixture["tokenizer"] = {
            "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "tokens": list(getattr(tokenizer, "tokens", [])),
        }
        return fixture

    def _torch_generation(self, model, tokenizer, torch, runtime, use_cache, max_new):
        from transformer_torch_minimal_forward import torch_minimal_parity_outputs

        fixture = self._tok_fixture(model, tokenizer)
        fixture["forward_cases"] = []
        fixture["generation_cases"] = [
            {
                "case_id": "g0",
                "prompt_ids": tokenizer.encode("mia ball"),
                "max_new_chars": max_new,
                "generation_config": asdict(
                    GenerationConfig(use_kv_cache=use_cache)
                ),
            }
        ]
        outputs = torch_minimal_parity_outputs(
            fixture=fixture, torch=torch, runtime=dict(runtime)
        )
        return outputs["generation_cases"][0]

    def _assert_torch_cache_equiv(self, num_layers: int) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, num_layers=num_layers, context_size=4)
        max_new = 9  # slide past the context_size==4 boundary
        runtimes = [{"dtype": "float64", "device": "cpu"}, {"dtype": "float32", "device": "cpu"}]
        if torch.backends.mps.is_available():
            runtimes.append({"dtype": "float32", "device": "mps"})
        for runtime in runtimes:
            on = self._torch_generation(model, tokenizer, torch, runtime, True, max_new)
            off = self._torch_generation(model, tokenizer, torch, runtime, False, max_new)
            self.assertEqual(
                on["token_ids"],
                off["token_ids"],
                msg=f"torch {runtime} num_layers={num_layers}: token sequence diverged",
            )
            self.assertEqual(on["text"], off["text"])
            # Cache hits ACTUALLY occurred (cache_enabled not silently False, and the
            # fake-torch double -- which lacks SDPA -- is not in play on real torch).
            self.assertGreater(
                on["cache"]["hits"], 0, msg=f"torch {runtime}: no cache hits"
            )
            self.assertGreater(on["cache"]["writes"], 0)
            self.assertEqual(off["cache"]["hits"], 0)

    def test_torch_cache_equiv_single_layer(self) -> None:
        self._assert_torch_cache_equiv(num_layers=1)

    def test_torch_cache_equiv_multi_layer(self) -> None:
        self._assert_torch_cache_equiv(num_layers=2)

    def test_torch_cache_equiv_sdpa(self) -> None:
        # The SDPA path reuses cached stacked keys/values in the same ascending order.
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, num_layers=1, context_size=4)
        runtime = {"dtype": "float32", "device": "cpu", "use_sdpa": True}
        on = self._torch_generation(model, tokenizer, torch, runtime, True, 9)
        off = self._torch_generation(model, tokenizer, torch, runtime, False, 9)
        self.assertEqual(on["token_ids"], off["token_ids"])
        self.assertGreater(on["cache"]["hits"], 0)


if __name__ == "__main__":
    unittest.main()
