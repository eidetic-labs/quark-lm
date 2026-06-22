"""Phase 1 parity for absolute-keyed RoPE (scalar + torch).

The contract: with ``use_absolute_rope`` OFF the model is byte-for-byte the
pre-absolute (enumerate-keyed) RoPE path on BOTH engines; with it ON, RoPE keys
consume the ABSOLUTE stream position of each window slot, a left-pad slot
(POSITION_PAD_SENTINEL = -1) rotates by the IDENTITY, position_embeddings and all
context summaries stay slot-keyed, and the scalar/torch engines agree within the
validated dtype band (float64 1e-6 / float32 3e-3) plus zero-tolerance rank order.

Three adversarially-confirmed disciplines are pinned here:
  * the scalar RoPE site GATES consumption on use_absolute_rope (test #1, exact ==),
  * the pad sentinel is a HARD-CODED identity, not trig (test #5, exact ==),
  * the batched Tier-2 path FAILS CLOSED to Tier-1 under the flag (test #12).

The padded-window difference is verified COMPONENT-WISE (test #6): the dominant
component is REAL tokens re-keyed from slot angles to absolute angles, with the pad
slots an exact identity -- NOT a single loose |on-off|>1e-9 threshold, which cannot
distinguish "sentinel fired" from a re-key bug.

Torch cases run only under real torch; system python3 silently skips them (a false
green), so run with ``PYTHONPATH=src:tests .venv/bin/python -m unittest``.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import asdict
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from neural_char_ops import (
    POSITION_PAD_SENTINEL,
    make_context,
    make_context_positioned,
)
from tokenizer import CharTokenizer
from transformer_model import GenerationConfig, TransformerConfig
from transformer_parity_contract import assert_numeric_parity, assert_rank_invariant
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_attention import _apply_rotary_row, torch_apply_rotary
from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_minimal_forward import torch_minimal_parity_outputs
from transformer_torch_profile_support import batched_forward_unsupported_reason

CORPUS = "mia ball box red cup noah shelf ava leo book\n"


def _model(tokenizer: CharTokenizer, **overrides) -> TinyTransformerLM:
    defaults = dict(
        vocab_size=tokenizer.vocab_size,
        context_size=6,
        embedding_dim=4,
        feedforward_dim=8,
        attention_heads=2,
        seed=7,
        use_rotary_positions=True,  # default-on profile: RoPE runs on the flag-off path
    )
    defaults.update(overrides)
    return TinyTransformerLM.init_random(TransformerConfig(**defaults))


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


def _fixture(model: TinyTransformerLM) -> dict:
    return {"weights": model.to_dict()["weights"], "model_config": asdict(model.config)}


def _tok_fixture(model: TinyTransformerLM, tokenizer: CharTokenizer) -> dict:
    # Match the runtime tokenizer summary the minimal-forward generation path reads
    # (top-level pad_id + tokens), not CharTokenizer.to_dict() which omits pad_id.
    fixture = _fixture(model)
    fixture["tokenizer"] = {
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "tokens": list(getattr(tokenizer, "tokens", [])),
    }
    return fixture


def _torch_logits(model, tokenizer, context, torch, runtime, *, abs_positions=None):
    fixture = _fixture(model)
    runtime = dict(runtime)
    if abs_positions is not None:
        runtime["abs_positions"] = abs_positions
    logits = torch_minimal_logits(context, fixture, torch, runtime)
    return [float(v) for v in logits.detach().cpu().tolist()]


CPU64 = {"dtype": "float64", "device": "cpu"}
CPU32 = {"dtype": "float32", "device": "cpu"}


class AbsoluteRopeScalarTest(unittest.TestCase):
    def test_flag_off_byte_exact_default(self) -> None:
        # verdict-1 guard: with the flag OFF, threading positions must be dropped at
        # the RoPE site, so logits are byte-IDENTICAL to the enumerate-keyed path and
        # to a multi-step short-prompt generation (padded positions like
        # [-1,-1,-1,0,1,2]). Exact ==, not tolerance.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer)  # flag off
        ids = tokenizer.encode("mia")
        context, positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        self.assertEqual(positions, [POSITION_PAD_SENTINEL] * 3 + [0, 1, 2])
        golden = model._forward_floats(context)  # pre-change: positions=None
        self.assertEqual(model._forward_floats(context, None), golden)
        self.assertEqual(model._forward_floats(context, positions), golden)
        self.assertEqual(model.predict(context, positions), model.predict(context))

        # Multi-step greedy generation from a SHORT prompt must be byte-identical to a
        # golden produced by the enumerate path. Build the golden by temporarily
        # forcing positions=None semantics: flag-off already does exactly that, so the
        # golden is a fresh model run -- and the assertion is that the live generation
        # (which threads positions) equals a run that cannot see them.
        gen = model.generate_with_trace(tokenizer, "mia", 5, GenerationConfig())
        # Reconstruct the enumerate-keyed golden by driving predict(context) (no
        # positions) through the identical greedy loop.
        golden_text, golden_ids = self._enumerate_generation(model, tokenizer, "mia", 5)
        self.assertEqual(gen["text"], golden_text)
        self.assertEqual([s["token_id"] for s in gen["trace"]], golden_ids)

    @staticmethod
    def _enumerate_generation(model, tokenizer, prompt, max_new):
        ids = tokenizer.encode(prompt)
        generated: list[int] = []
        token_ids: list[int] = []
        for _ in range(max_new):
            context = make_context(ids, model.config.context_size, tokenizer.pad_id)
            probs = model.predict(context)  # positions=None -> enumerate
            next_id = max(range(len(probs)), key=lambda index: probs[index])
            token_ids.append(next_id)
            ids.append(next_id)
            generated.append(next_id)
        return tokenizer.decode(generated), token_ids

    def test_unpadded_window_flag_on_equals_off_f64(self) -> None:
        # Contiguous (all-real) window: absolute positions == enumerate, so flag-on
        # must equal flag-off at f64 1e-6 with identical rank order.
        tokenizer = CharTokenizer.train(CORPUS)
        off = _model(tokenizer)
        on = _model(tokenizer, use_absolute_rope=True)
        ids = tokenizer.encode("mia ball box red cup")  # >= context_size 6
        context, positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        self.assertTrue(all(p >= 0 for p in positions))  # no pads
        self.assertEqual(positions, list(range(positions[0], positions[0] + 6)))
        off_logits = off._forward_floats(context)
        on_logits = on._forward_floats(context, positions)
        assert_numeric_parity(off_logits, on_logits, dtype="float64")
        assert_rank_invariant(off_logits, on_logits)

    def test_constant_shift_cancels_f64(self) -> None:
        # RoPE shift-invariance: a constant offset on every (non-pad) position cancels
        # in the q.k inner product over a contiguous window.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        context = make_context(tokenizer.encode("mia ball box red cup"), 6, tokenizer.pad_id)
        base = list(range(0, 6))
        ref = model._forward_floats(context, base)
        for shift in (1, 1000):
            shifted = [p + shift for p in base]
            assert_numeric_parity(ref, model._forward_floats(context, shifted), dtype="float64")

    def test_pad_slot_is_identity_rotation(self) -> None:
        # verdict-2 part A: a pad slot (-1) rotates by EXACT identity (q/k unrotated).
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer)
        from autograd import Scalar

        row_scalars = [Scalar(0.3), Scalar(-0.7), Scalar(1.1), Scalar(-0.2)]
        rotated = model._apply_rotary_scalars([row_scalars], positions=[POSITION_PAD_SENTINEL])
        self.assertEqual([s.data for s in rotated[0]], [s.data for s in row_scalars])
        row_floats = [0.3, -0.7, 1.1, -0.2]
        rotated_f = model._apply_rotary_floats([row_floats], positions=[POSITION_PAD_SENTINEL])
        self.assertEqual(rotated_f[0], row_floats)

    def test_padded_window_difference_is_rekey_plus_pad_identity(self) -> None:
        # verdict-2 part B: decompose flag-on minus flag-off on a PADDED window
        # COMPONENT-WISE at the rotary level (the only thing that changes):
        #   (a) real tokens are re-keyed from slot angles to absolute angles,
        #   (b) pad slots are an exact identity.
        # The dominant component is the real-token re-keying, NOT the pad sentinel.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        ids = tokenizer.encode("ab")  # short -> heavy left-pad
        context, positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        # positions like [-1,-1,-1,-1,0,1] vs enumerate [0,1,2,3,4,5]
        self.assertEqual(positions, [POSITION_PAD_SENTINEL] * 4 + [0, 1])
        from autograd import Scalar

        rng_rows = [
            [Scalar(0.5), Scalar(-0.3), Scalar(0.9), Scalar(-1.2)],
            [Scalar(-0.4), Scalar(0.8), Scalar(-0.1), Scalar(0.6)],
            [Scalar(1.0), Scalar(-0.5), Scalar(0.2), Scalar(-0.9)],
            [Scalar(-0.7), Scalar(0.1), Scalar(0.4), Scalar(-0.6)],
            [Scalar(0.3), Scalar(0.7), Scalar(-0.8), Scalar(0.2)],
            [Scalar(-0.2), Scalar(-0.6), Scalar(1.1), Scalar(0.5)],
        ]
        absolute = model._apply_rotary_scalars(rng_rows, positions=positions)
        slot = model._apply_rotary_scalars(rng_rows, positions=None)  # enumerate

        head_dim = model.config.embedding_dim // model.config.attention_heads

        def expected_row(row, position):
            # Hand-roll the documented decomposition for one row.
            output = [s.data for s in row]
            for head in range(model.config.attention_heads):
                start = head * head_dim
                for offset in range(0, head_dim - 1, 2):
                    index = start + offset
                    if position < 0:
                        cos_v, sin_v = 1.0, 0.0  # pad slot: identity, not trig
                    else:
                        angle = position / (10000.0 ** (offset / max(head_dim, 1)))
                        cos_v, sin_v = math.cos(angle), math.sin(angle)
                    left, right = row[index].data, row[index + 1].data
                    output[index] = left * cos_v - right * sin_v
                    output[index + 1] = left * sin_v + right * cos_v
            return output

        # (a)+(b): the absolute result matches the component-wise expected EXACTLY.
        for i, position in enumerate(positions):
            assert_numeric_parity(
                [s.data for s in absolute[i]], expected_row(rng_rows[i], position), dtype="float64"
            )
        # Pad rows (position < 0) are exact identity AND identical between slot/abs
        # (slot enumerate at i is also >=0, so the pad rows differ from the slot path
        # only by being unrotated) -- assert the pad rows equal their raw input.
        for i, position in enumerate(positions):
            if position < 0:
                self.assertEqual(
                    [s.data for s in absolute[i]], [s.data for s in rng_rows[i]]
                )
        # Dominant component is the real-token re-keying: at least one real slot
        # (position >= 0 but != its enumerate index) genuinely changed.
        real_changed = any(
            position >= 0
            and position != i
            and [s.data for s in absolute[i]] != [s.data for s in slot[i]]
            for i, position in enumerate(positions)
        )
        self.assertTrue(real_changed, "expected real tokens re-keyed slot->absolute")

    def test_odd_head_dim(self) -> None:
        # embedding_dim 6 / 2 heads -> head_dim 3 (odd): the tail dim is untouched and
        # offset=0 gives angle == raw position. Flag-on full window == flag-off f64;
        # pad slot exact identity.
        tokenizer = CharTokenizer.train(CORPUS)
        off = _model(tokenizer, embedding_dim=6, attention_heads=2, feedforward_dim=8)
        on = _model(
            tokenizer, embedding_dim=6, attention_heads=2, feedforward_dim=8,
            use_absolute_rope=True,
        )
        self.assertEqual(off.config.embedding_dim // off.config.attention_heads, 3)
        ids = tokenizer.encode("mia ball box red cup")
        context, positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        off_logits = off._forward_floats(context)
        on_logits = on._forward_floats(context, positions)
        assert_numeric_parity(off_logits, on_logits, dtype="float64")
        assert_rank_invariant(off_logits, on_logits)
        from autograd import Scalar

        row = [Scalar(0.4), Scalar(-0.5), Scalar(0.9), Scalar(-0.2), Scalar(0.7), Scalar(-0.1)]
        pad = on._apply_rotary_scalars([row], positions=[POSITION_PAD_SENTINEL])
        self.assertEqual([s.data for s in pad[0]], [s.data for s in row])

    def test_multi_layer(self) -> None:
        # num_layers > 1 exercises _forward_full_block_* threading AND the
        # freeze-lower-layers float branch in _final_hidden_scalars.
        tokenizer = CharTokenizer.train(CORPUS)
        off = _model(tokenizer, num_layers=2)
        on = _model(tokenizer, num_layers=2, use_absolute_rope=True)
        ids = tokenizer.encode("mia ball box red cup")
        context, positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        off_logits = off._forward_floats(context)
        on_logits = on._forward_floats(context, positions)
        assert_numeric_parity(off_logits, on_logits, dtype="float64")
        assert_rank_invariant(off_logits, on_logits)

        # Flag-off (scalar) byte-exact vs the enumerate golden on a padded window.
        pad_ids = tokenizer.encode("mia")
        pad_context, pad_positions = make_context_positioned(pad_ids, 6, tokenizer.pad_id)
        golden = off._forward_scalars(pad_context)
        threaded = off._forward_scalars(pad_context, pad_positions)
        self.assertEqual([s.data for s in threaded], [s.data for s in golden])

        # Freeze-lower-layers float branch: thread positions through it too.
        on.freeze_lower_layers_for_updates = True
        frozen = on._forward_scalars(context, positions)
        on.freeze_lower_layers_for_updates = False
        unfrozen = on._forward_scalars(context, positions)
        assert_numeric_parity(
            [s.data for s in frozen], [s.data for s in unfrozen], dtype="float64"
        )

    def test_generation_threads_positions(self) -> None:
        # Spy: generate_with_trace builds make_context_positioned and threads positions
        # of length context_size into predict.
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        seen: list[tuple[list[int], list[int] | None]] = []
        original_predict = model.predict

        def spy_predict(context, positions=None):
            seen.append((list(context), None if positions is None else list(positions)))
            return original_predict(context, positions)

        model.predict = spy_predict  # type: ignore[method-assign]
        model.generate_with_trace(tokenizer, "mia", 3, GenerationConfig())
        self.assertTrue(seen)
        for context, positions in seen:
            self.assertIsNotNone(positions)
            self.assertEqual(len(positions), model.config.context_size)
            self.assertEqual(len(context), model.config.context_size)
        # First step: short prompt -> leading pad sentinels then absolute indices.
        self.assertEqual(seen[0][1], [POSITION_PAD_SENTINEL] * 3 + [0, 1, 2])

    def test_batched_fails_closed_under_absolute_rope(self) -> None:
        # verdict-3 guard: the Tier-2 batched forward must route to Tier-1 whenever the
        # flag is on, so it can never silently run slot-keyed while scalar runs absolute.
        reason = batched_forward_unsupported_reason({"use_absolute_rope": True})
        self.assertIsNotNone(reason)
        self.assertIsNone(batched_forward_unsupported_reason({}))


class AbsoluteRopeTorchTest(unittest.TestCase):
    def test_flag_off_byte_exact_torch_generation(self) -> None:
        # Guards that swapping the duplicate _make_context for make_context_positioned
        # did not perturb the flag-off torch generation path: byte-identical to a
        # golden produced by the scalar enumerate generation on the same model.
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer)  # flag off
        fixture = _tok_fixture(model, tokenizer)
        fixture["forward_cases"] = []
        fixture["generation_cases"] = [
            {
                "case_id": "g0",
                "prompt_ids": tokenizer.encode("mia"),
                "max_new_chars": 5,
                "generation_config": asdict(GenerationConfig()),
            }
        ]
        runtime = dict(CPU64)
        outputs = torch_minimal_parity_outputs(fixture=fixture, torch=torch, runtime=runtime)
        gen = outputs["generation_cases"][0]
        golden_text, golden_ids = AbsoluteRopeScalarTest._enumerate_generation(
            model, tokenizer, "mia", 5
        )
        self.assertEqual(gen["text"], golden_text)
        self.assertEqual(gen["token_ids"], golden_ids)

    def test_scalar_torch_lockstep_unpadded_f64(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        ids = tokenizer.encode("mia ball box red cup")
        context, positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        scalar = model._forward_floats(context, positions)
        torch_vals = _torch_logits(model, tokenizer, context, torch, CPU64, abs_positions=positions)
        assert_numeric_parity(scalar, torch_vals, dtype="float64")
        assert_rank_invariant(scalar, torch_vals)

    def test_scalar_torch_lockstep_f32_mps(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        ids = tokenizer.encode("mia ball box red cup")
        context, positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        scalar = model._forward_floats(context, positions)
        for runtime in (CPU32,) + (
            ({"dtype": "float32", "device": "mps"},) if torch.backends.mps.is_available() else ()
        ):
            torch_vals = _torch_logits(
                model, tokenizer, context, torch, runtime, abs_positions=positions
            )
            assert_numeric_parity(scalar, torch_vals, dtype="float32")
            assert_rank_invariant(scalar, torch_vals)

    def test_torch_pad_slot_is_identity_rotation(self) -> None:
        # The torch single-query _apply_rotary_row identity branch is exact, including
        # on f32/MPS (hard-coded cos=1.0/sin=0.0, not device trig).
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer)
        config = asdict(model.config)
        for runtime in (CPU64, CPU32):
            row = torch.tensor([0.3, -0.7, 1.1, -0.2], dtype=getattr(torch, runtime["dtype"]))
            out = _apply_rotary_row(row, POSITION_PAD_SENTINEL, config, torch)
            self.assertEqual(
                [float(v) for v in out.detach().cpu().tolist()],
                [float(v) for v in row.detach().cpu().tolist()],
            )
            # And via the row-vectorized entry point with an all-pad position list.
            stacked = torch.stack([row])
            rotated = torch_apply_rotary(stacked, config, torch, [POSITION_PAD_SENTINEL])
            self.assertEqual(
                [float(v) for v in rotated[0].detach().cpu().tolist()],
                [float(v) for v in row.detach().cpu().tolist()],
            )


if __name__ == "__main__":
    unittest.main()
