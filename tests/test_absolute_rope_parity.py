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

import json
import math
import tempfile
import unittest
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import support  # noqa: F401  (puts src/ on sys.path)
import transformer_torch_contrast as contrast_module
from answer_examples import AnswerExample
from neural_char_ops import (
    POSITION_PAD_SENTINEL,
    make_context,
    make_context_positioned,
)
from tokenizer import CharTokenizer
from transformer_model import GenerationConfig, OptimizationConfig, TransformerConfig
from transformer_parity_contract import assert_numeric_parity, assert_rank_invariant
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_attention import _apply_rotary_row, torch_apply_rotary
from transformer_torch_contrast import (
    torch_answer_choice_loss,
    torch_answer_sequence_loss,
    train_torch_answer_mixed,
)
from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_minimal_forward import torch_minimal_parity_outputs
from transformer_torch_profile_support import batched_forward_unsupported_reason
from transformer_torch_training_loop import save_torch_checkpoint
from transformer_torch_training_loss import (
    build_torch_training_logits,
    build_torch_training_loss_tensor,
)
from transformer_torch_training_state import build_torch_training_state
from transformer_training_parity_fixture import build_scalar_training_parity_fixture

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


def _zero_position_embeddings(model: TinyTransformerLM) -> None:
    """Zero the learned position_embeddings so a flag-OFF model adds no positional addend.

    Phase 2 drops the learned pos-embed addend under use_absolute_rope, so an ON forward
    no longer equals an OFF forward that still adds the table. Zeroing the OFF table
    re-grounds the comparison: OFF-zeroed contributes token embeddings only, exactly like
    the ON arm, isolating the RoPE keying as the sole remaining difference.
    """

    for row in model.position_embeddings:
        for scalar in row:
            scalar.data = 0.0


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
        # Contiguous (all-real) window: absolute positions == enumerate. Phase 2 drops the
        # learned pos-embed addend under the flag, so flag-on no longer equals a flag-off
        # that STILL adds pos-embed. Re-ground: zero the OFF model's position_embeddings so
        # OFF also contributes no positional addend, then ON (token-only + absolute RoPE)
        # must equal OFF-zeroed (token-only + slot RoPE == absolute RoPE on a contiguous
        # window) at f64 1e-6 with identical rank order.
        tokenizer = CharTokenizer.train(CORPUS)
        off = _model(tokenizer)
        _zero_position_embeddings(off)
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
        # Phase 2: re-ground ON==OFF-zeroed (OFF still adds pos-embed otherwise).
        _zero_position_embeddings(off)
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
        # Phase 2: re-ground the unpadded ON==OFF on a SEPARATE pos-embed-zeroed OFF model
        # (token-only + slot RoPE == ON's token-only + absolute RoPE on a contiguous
        # window). The original `off` stays intact for the byte-exact padded check below.
        off_zeroed = _model(tokenizer, num_layers=2)
        _zero_position_embeddings(off_zeroed)
        off_logits = off_zeroed._forward_floats(context)
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

        def spy_predict(context, positions=None, cache=None):
            # Phase 3 added the optional ``cache`` param to predict; mirror the live
            # signature so generate_with_trace's call (context, positions, kv_cache)
            # binds. The flag-off generation here passes cache=None.
            seen.append((list(context), None if positions is None else list(positions)))
            return original_predict(context, positions, cache)

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


CONTRAST_PAIRS = [
    (AnswerExample("mia ball", " a", "f"), AnswerExample("noah ball", " u", "o")),
    (AnswerExample("noah cup", " b", "f"), AnswerExample("mia cup", " u", "o")),
]
CONTRAST_TEXT = (
    "".join(e.prompt + e.target for pair in CONTRAST_PAIRS for e in pair) + "\n"
)


def _abs_fixture(model: TinyTransformerLM, tokenizer: CharTokenizer) -> dict:
    """A training-parity fixture carrying the tokenizer summary the training path reads."""

    context = make_context(tokenizer.encode("mia ball"), model.config.context_size, tokenizer.pad_id)
    fixture = build_scalar_training_parity_fixture(
        fixture_id="abs-train",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=tokenizer.encode(" a")[0],
        optimizer_config=OptimizationConfig(
            optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
        ),
        learning_rate=0.02,
        steps=1,
        corpus_hash="abs",
    )
    fixture["tokenizer"] = {
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "tokens": list(getattr(tokenizer, "tokens", [])),
    }
    return fixture


class AbsoluteRopePhase2TrainingTest(unittest.TestCase):
    """Phase 2: the torch TRAINING path keys RoPE absolutely (not slot-keyed).

    Driven through the REAL training entry (build_torch_training_logits /
    torch_answer_sequence_loss / train_torch_answer_mixed), NOT the test-only
    _torch_logits helper -- the helper injects abs_positions directly and so cannot prove
    the training context-build threads them.
    """

    def test_torch_training_path_keys_absolute_not_slot(self) -> None:
        # R-A guard. On a PADDED context (left-pad -> absolute != enumerate), the training
        # path's logits must equal the scalar ABSOLUTE reference and DIFFER from the scalar
        # SLOT-keyed reference. If the training path slot-keyed, it would match the latter.
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)  # _model sets use_rotary_positions
        fixture = _abs_fixture(model, tokenizer)
        state = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU64)
        ids = tokenizer.encode("mia")  # short -> heavy left-pad
        context, abs_positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        self.assertIn(POSITION_PAD_SENTINEL, abs_positions)  # genuinely padded

        step_runtime = dict(CPU64)
        step_runtime["abs_positions"] = abs_positions
        training_logits = build_torch_training_logits(
            fixture=fixture, state=state, torch=torch, runtime=step_runtime, context=context
        )
        training_vals = [float(v) for v in training_logits.detach().cpu().tolist()]

        scalar_absolute = model._forward_floats(context, abs_positions)
        scalar_slot = model._forward_floats(context, list(range(6)))  # enumerate keying
        assert_numeric_parity(scalar_absolute, training_vals, dtype="float64")
        # And the slot-keyed scalar reference is genuinely DIFFERENT on this padded window,
        # so the parity above is load-bearing (not a degenerate equality).
        max_slot_gap = max(abs(a - b) for a, b in zip(scalar_slot, training_vals))
        self.assertGreater(max_slot_gap, 1e-9)

    def test_next_token_term_uses_absolute(self) -> None:
        # R-A's specific missed site: the next-token term of train_torch_answer_mixed calls
        # build_torch_training_loss_tensor DIRECTLY (bypassing _batch_loss). A few steps on
        # a left-padded triple must NOT raise the fail-closed guard, and the loss tensor at
        # a padded context must equal the scalar ABSOLUTE reference loss (not slot-keyed).
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CONTRAST_TEXT)
        context_size = 12  # > prompt length so the early windows carry left-pad
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size, context_size=context_size, embedding_dim=4,
            feedforward_dim=8, seed=11, use_rotary_positions=True, use_absolute_rope=True,
        )
        model = TinyTransformerLM.init_random(config)
        fixture = _abs_fixture(model, tokenizer)
        # Build left-padded triples for the next-token term.
        examples = []
        for in_example, _ooc in CONTRAST_PAIRS:
            tids = list(tokenizer.encode(in_example.prompt))
            for target_id in tokenizer.encode(in_example.target):
                context, abs_positions = make_context_positioned(tids, context_size, tokenizer.pad_id)
                examples.append((context, abs_positions, target_id))
                tids.append(target_id)
        # The first emitted window is genuinely left-padded (absolute != enumerate).
        self.assertIn(POSITION_PAD_SENTINEL, examples[0][1])

        # A few steps train without tripping the fail-closed guard (proves abs_positions
        # are threaded at the direct build_torch_training_loss_tensor call site).
        state, losses = train_torch_answer_mixed(
            fixture=fixture, tokenizer=tokenizer, examples=examples,
            contrast_pairs=CONTRAST_PAIRS, steps=4, learning_rate=0.02, contrast_weight=1.0,
            torch=torch, runtime=CPU64,
        )
        self.assertEqual(len(losses), 4)
        self.assertTrue(all(value == value for value in losses))  # all finite

        # The next-token loss at a padded context, threaded with abs_positions, equals the
        # scalar ABSOLUTE reference loss; a slot-keyed forward would not.
        untrained = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU64)
        context, abs_positions, target = examples[0]
        step_runtime = dict(CPU64)
        step_runtime["abs_positions"] = abs_positions
        torch_loss = float(
            build_torch_training_loss_tensor(
                fixture=fixture, state=untrained, torch=torch, runtime=step_runtime,
                context=context, target=target,
            ).detach().cpu()
        )
        scalar_loss = model.nll(context, target, abs_positions)
        self.assertAlmostEqual(torch_loss, scalar_loss, delta=1e-6)

    def test_both_contrast_contexts_get_independent_absolute_positions(self) -> None:
        # Spy on make_context_positioned within the contrast objective: both the owner
        # (in_example) and the entity-swapped non-owner (ooc_example) prompt streams must
        # produce their OWN abs_positions (correct length, derived from their own ids), and
        # the SHARED runtime dict must be byte-identical before and after (no mutation --
        # the per-call copy lives inside the per-target loop).
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CONTRAST_TEXT)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size, context_size=8, embedding_dim=4,
            feedforward_dim=8, seed=11, use_rotary_positions=True, use_absolute_rope=True,
        )
        model = TinyTransformerLM.init_random(config)
        fixture = _abs_fixture(model, tokenizer)
        state = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU64)

        in_example, ooc_example = CONTRAST_PAIRS[0]
        seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        original = contrast_module.make_context_positioned

        def spy(ids, context_size, pad_id):
            context, positions = original(ids, context_size, pad_id)
            seen.append((tuple(ids), tuple(positions)))
            return context, positions

        shared_runtime = dict(CPU64)
        before = dict(shared_runtime)
        contrast_module.make_context_positioned = spy
        try:
            torch_answer_choice_loss(
                fixture=fixture, state=state, prompt_ids=tokenizer.encode(in_example.prompt),
                candidate_token_lists=[tokenizer.encode(in_example.target), tokenizer.encode(ooc_example.target)],
                torch=torch, runtime=shared_runtime,
            )
            torch_answer_choice_loss(
                fixture=fixture, state=state, prompt_ids=tokenizer.encode(ooc_example.prompt),
                candidate_token_lists=[tokenizer.encode(ooc_example.target), tokenizer.encode(in_example.target)],
                torch=torch, runtime=shared_runtime,
            )
        finally:
            contrast_module.make_context_positioned = original

        self.assertTrue(seen)
        # Every produced positions list has length context_size (8).
        for _ids, positions in seen:
            self.assertEqual(len(positions), 8)
        # The owner and non-owner prompts have DIFFERENT entities, so at least two distinct
        # id-streams were observed (independent prompt streams -> independent positions).
        distinct_streams = {ids for ids, _positions in seen}
        self.assertGreaterEqual(len(distinct_streams), 2)
        # The shared runtime dict was NOT mutated (per-call dict(runtime) copy discipline;
        # pins the runtime-report parity check).
        self.assertEqual(shared_runtime, before)
        self.assertNotIn("abs_positions", shared_runtime)

    def test_scalar_equals_torch_training_path_under_flag_padded(self) -> None:
        # scalar _forward_floats(context, positions) == torch TRAINING-path logits at f64 on
        # a PADDED window (drop-posembed parity through the real training entry).
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        fixture = _abs_fixture(model, tokenizer)
        state = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU64)
        ids = tokenizer.encode("noah")  # short -> left-pad
        context, abs_positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        self.assertIn(POSITION_PAD_SENTINEL, abs_positions)
        step_runtime = dict(CPU64)
        step_runtime["abs_positions"] = abs_positions
        torch_vals = [
            float(v)
            for v in build_torch_training_logits(
                fixture=fixture, state=state, torch=torch, runtime=step_runtime, context=context
            ).detach().cpu().tolist()
        ]
        scalar = model._forward_floats(context, abs_positions)
        assert_numeric_parity(scalar, torch_vals, dtype="float64")
        assert_rank_invariant(scalar, torch_vals)

    def test_fail_closed_guard_raises_without_abs_positions(self) -> None:
        # R-C: torch_minimal_logits under use_absolute_rope with a runtime MISSING
        # abs_positions must RAISE (a missed training site crashes, not silently
        # slot-keys). OFF (no flag) with the same missing-positions runtime is fine.
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        fixture = _fixture(model)
        context = make_context(tokenizer.encode("mia"), 6, tokenizer.pad_id)
        with self.assertRaises(ValueError):
            torch_minimal_logits(context, fixture, torch, dict(CPU64))  # no abs_positions

        off_model = _model(tokenizer)  # flag off
        off_fixture = _fixture(off_model)
        torch_minimal_logits(context, off_fixture, torch, dict(CPU64))  # must not raise

    def test_checkpoint_loads_and_round_trips_flag(self) -> None:
        # Format stays quarklm-transformer-v2, position_embeddings key present + correct
        # shape, use_absolute_rope round-trips True; and an OFF-trained checkpoint loads
        # into an ON model (A/B contract) and replays.
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(CORPUS)
        model = _model(tokenizer, use_absolute_rope=True)
        fixture = _abs_fixture(model, tokenizer)
        ids = tokenizer.encode("mia")
        context, abs_positions = make_context_positioned(ids, 6, tokenizer.pad_id)
        state = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU64)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "abs_answer.json"
            save_torch_checkpoint(path, fixture=fixture, state=state, tokenizer=tokenizer)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["checkpoint_format"], "quarklm-transformer-v2")
            self.assertIn("position_embeddings", payload["weights"])
            self.assertEqual(len(payload["weights"]["position_embeddings"]), 6)  # context_size rows
            self.assertTrue(payload["config"]["use_absolute_rope"])
            reloaded, reloaded_tokenizer = TinyTransformerLM.load(path)
            self.assertIsNotNone(reloaded_tokenizer)
            self.assertTrue(reloaded.config.use_absolute_rope)
            # Reloaded ON model replays RoPE-only at f64 vs the original ON forward.
            assert_numeric_parity(
                model._forward_floats(context, abs_positions),
                reloaded._forward_floats(context, abs_positions),
                dtype="float64",
            )

        # An OFF checkpoint loads into an ON model (A/B footgun documented in the plan):
        # from_dict honors the saved config, so an OFF ckpt reconstructs an OFF model.
        off_model = _model(tokenizer)
        payload_off = off_model.to_dict(tokenizer)
        reloaded_off, _tok = TinyTransformerLM.from_dict(payload_off)
        self.assertFalse(reloaded_off.config.use_absolute_rope)

    def test_use_rotary_positions_guard_rejects_position_blind_config(self) -> None:
        # R-B at config level: use_absolute_rope=True WITHOUT use_rotary_positions=True is a
        # position-blind config (no pos-embed AND no RoPE) and must be refused at
        # construction; with use_rotary_positions=True it constructs.
        tokenizer = CharTokenizer.train(CORPUS)
        with self.assertRaises(ValueError):
            TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=tokenizer.vocab_size, context_size=6, embedding_dim=4,
                    feedforward_dim=8, attention_heads=2, seed=7,
                    use_absolute_rope=True, use_rotary_positions=False,
                )
            )
        TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size, context_size=6, embedding_dim=4,
                feedforward_dim=8, attention_heads=2, seed=7,
                use_absolute_rope=True, use_rotary_positions=True,
            )
        )  # must not raise


if __name__ == "__main__":
    unittest.main()
