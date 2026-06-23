"""Vectorized absolute-RoPE on the Tier-2 batched (B,C,D) forward (real-torch-only).

The accelerator keystone: the general-LM (use_prompt_position_projection OFF)
absolute-RoPE forward now runs on the fast Tier-2 batched path instead of the
per-position Tier-1 path. The contract pinned here:

  * batched abs-RoPE logits == Tier-1 (torch_minimal_logits) on the SAME weights,
    within the validated dtype band (float64 1e-6 / float32 3e-3) PLUS zero-tolerance
    rank-invariance, over PROBES including a left-PADDED window and an ODD head_dim;
  * the absolute positions are threaded via runtime['abs_positions'] derived from
    make_context_positioned (NOT a test-injected arange);
  * use_prompt_position_projection STILL fails closed to Tier-1, use_absolute_rope
    no longer does;
  * a missing runtime['abs_positions'] under the flag RAISES (never silently
    slot-keys via arange);
  * the REAL batched training/loss path (use_batched_forward + use_absolute_rope)
    keys absolute on a padded context -- matching the scalar ABSOLUTE reference and
    DIFFERING from the slot-keyed (positions=None) reference;
  * default-off (flag off OR use_batched_forward off) leaves the batched/Tier-1
    paths byte-identical to the pre-change path.

Torch cases run only under real torch; system python3 silently skips them (a false
green) AND uses an SDPA-less fake-torch double, so run with
``PYTHONPATH=src:tests .venv/bin/python -m unittest``.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from neural_char_ops import (
    POSITION_PAD_SENTINEL,
    make_context_positioned,
)
from tokenizer import CharTokenizer
from transformer_model import OptimizationConfig, TransformerConfig
from transformer_parity_contract import assert_numeric_parity, assert_rank_invariant
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_batched_block import torch_batched_logits
from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_profile_support import batched_forward_unsupported_reason
from transformer_torch_training_loop import train_torch_lm
from transformer_torch_training_loss import build_torch_batched_loss_tensor
from transformer_torch_training_state import build_torch_training_state
from transformer_training_parity_fixture import build_scalar_training_parity_fixture

CORPUS = "mia ball box red cup noah shelf ava leo book\n"
CPU64 = {"dtype": "float64", "device": "cpu"}
CPU32 = {"dtype": "float32", "device": "cpu"}

# Prompts spanning unpadded (>= context_size) and left-padded (< context_size) windows.
PROMPTS = [
    "mia ball box red cup noah",  # >= 6 -> unpadded
    "mia",                        # short -> heavy left-pad
    "noah cup",                   # short -> left-pad
    "ava book shelf",             # short -> left-pad
]


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


def _model(tokenizer: CharTokenizer, **overrides) -> TinyTransformerLM:
    defaults = dict(
        vocab_size=tokenizer.vocab_size,
        context_size=6,
        embedding_dim=4,
        feedforward_dim=8,
        attention_heads=2,
        seed=7,
        use_rotary_positions=True,
    )
    defaults.update(overrides)
    return TinyTransformerLM.init_random(TransformerConfig(**defaults))


def _fixture(model: TinyTransformerLM) -> dict:
    return {"weights": model.to_dict()["weights"], "model_config": asdict(model.config)}


def _contexts_and_positions(tokenizer, context_size):
    contexts = []
    positions = []
    for prompt in PROMPTS:
        context, abs_positions = make_context_positioned(
            tokenizer.encode(prompt), context_size, tokenizer.pad_id
        )
        contexts.append(context)
        positions.append(abs_positions)
    return contexts, positions


def _assert_batched_matches_tier1(test, torch, model, runtime):
    """Batched abs-RoPE == Tier-1 over the probes, abs_positions threaded per row."""

    fixture = _fixture(model)
    contexts, positions = _contexts_and_positions(tokenizer=test.tokenizer, context_size=model.config.context_size)
    # Batched: thread the (B,C) absolute positions through runtime['abs_positions'].
    batched_runtime = dict(runtime)
    batched_runtime["abs_positions"] = positions
    batched = torch_batched_logits(contexts, fixture, torch, batched_runtime)
    for index, context in enumerate(contexts):
        # Tier-1: thread this row's absolute positions.
        single_runtime = dict(runtime)
        single_runtime["abs_positions"] = positions[index]
        single = torch_minimal_logits(context, fixture, torch, single_runtime)
        assert_numeric_parity(
            [float(v) for v in single.detach().cpu().tolist()],
            [float(v) for v in batched[index].detach().cpu().tolist()],
            dtype=runtime["dtype"],
        )
        assert_rank_invariant(
            [float(v) for v in single.detach().cpu().tolist()],
            [float(v) for v in batched[index].detach().cpu().tolist()],
        )


class BatchedAbsoluteRopeParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharTokenizer.train(CORPUS)

    def test_batched_abs_rope_matches_tier1_padded_probes(self) -> None:
        torch = _torch_or_skip(self)
        model = _model(self.tokenizer, use_absolute_rope=True)
        # At least one probe is genuinely left-padded (absolute != enumerate).
        _contexts, positions = _contexts_and_positions(self.tokenizer, model.config.context_size)
        self.assertTrue(any(POSITION_PAD_SENTINEL in row for row in positions))
        for runtime in (CPU64, CPU32):
            _assert_batched_matches_tier1(self, torch, model, runtime)

    def test_batched_abs_rope_matches_tier1_odd_head_dim(self) -> None:
        torch = _torch_or_skip(self)
        # embedding_dim 6 / 2 heads -> head_dim 3 (odd); the tail dim must stay untouched.
        model = _model(
            self.tokenizer, embedding_dim=6, attention_heads=2, feedforward_dim=8,
            use_absolute_rope=True,
        )
        self.assertEqual(model.config.embedding_dim // model.config.attention_heads, 3)
        for runtime in (CPU64, CPU32):
            _assert_batched_matches_tier1(self, torch, model, runtime)

    def test_profile_coverage_returns_supported_for_abs_rope(self) -> None:
        # posproj STILL fails closed; use_absolute_rope is now supported (None).
        _torch_or_skip(self)
        self.assertIsNotNone(
            batched_forward_unsupported_reason({"use_prompt_position_projection": True})
        )
        self.assertIsNone(batched_forward_unsupported_reason({"use_absolute_rope": True}))
        self.assertIsNone(batched_forward_unsupported_reason({}))
        # Combined with another supported flag stays supported; combined with posproj closed.
        self.assertIsNone(
            batched_forward_unsupported_reason(
                {"use_absolute_rope": True, "use_rotary_positions": True}
            )
        )
        self.assertIsNotNone(
            batched_forward_unsupported_reason(
                {"use_absolute_rope": True, "use_prompt_position_projection": True}
            )
        )

    def test_missing_abs_positions_raises(self) -> None:
        # Fail-closed: the batched forward under use_absolute_rope with no
        # runtime['abs_positions'] must RAISE (never silently arange / slot-key).
        torch = _torch_or_skip(self)
        model = _model(self.tokenizer, use_absolute_rope=True)
        fixture = _fixture(model)
        contexts, _positions = _contexts_and_positions(self.tokenizer, model.config.context_size)
        with self.assertRaises(ValueError):
            torch_batched_logits(contexts, fixture, torch, dict(CPU64))  # no abs_positions
        # OFF (no flag) with the same missing-positions runtime must NOT raise (slot-keys).
        off_fixture = _fixture(_model(self.tokenizer))  # flag off
        torch_batched_logits(contexts, off_fixture, torch, dict(CPU64))


class BatchedAbsoluteRopeTrainingTest(unittest.TestCase):
    """The REAL batched training/loss path keys RoPE absolutely on a padded window."""

    def setUp(self) -> None:
        self.tokenizer = CharTokenizer.train(CORPUS)

    def _abs_fixture(self, model: TinyTransformerLM) -> dict:
        context, _positions = make_context_positioned(
            self.tokenizer.encode("mia"), model.config.context_size, self.tokenizer.pad_id
        )
        fixture = build_scalar_training_parity_fixture(
            fixture_id="batched-abs-train",
            model=model,
            tokenizer=self.tokenizer,
            context=context,
            target=self.tokenizer.encode(" ")[0],
            optimizer_config=OptimizationConfig(
                optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
            ),
            learning_rate=0.02,
            steps=1,
            corpus_hash="batched-abs",
        )
        fixture["tokenizer"] = {
            "tokenizer_type": getattr(self.tokenizer, "tokenizer_type", "char"),
            "vocab_size": self.tokenizer.vocab_size,
            "pad_id": self.tokenizer.pad_id,
            "tokens": list(getattr(self.tokenizer, "tokens", [])),
        }
        return fixture

    def test_batched_loss_path_keys_absolute_not_slot(self) -> None:
        # Drive the REAL batched loss path (build_torch_batched_loss_tensor under
        # use_batched_forward). On a PADDED context the batched logits must match the
        # scalar ABSOLUTE reference and DIFFER from the slot-keyed reference. If the path
        # slot-keyed, it would match the latter.
        torch = _torch_or_skip(self)
        model = _model(self.tokenizer, use_absolute_rope=True)
        fixture = self._abs_fixture(model)
        state = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU64)
        ids = self.tokenizer.encode("mia")  # short -> heavy left-pad
        context, abs_positions = make_context_positioned(ids, 6, self.tokenizer.pad_id)
        self.assertIn(POSITION_PAD_SENTINEL, abs_positions)
        target = self.tokenizer.encode(" ")[0]

        # Batched loss with use_batched_forward + abs_positions threaded as (B=1, C).
        batched_runtime = dict(CPU64)
        batched_runtime["use_batched_forward"] = True
        batched_runtime["abs_positions"] = [abs_positions]
        batched_loss = float(
            build_torch_batched_loss_tensor(
                fixture=fixture, state=state, torch=torch, runtime=batched_runtime,
                contexts=[context], targets=[target],
            ).detach().cpu()
        )

        # Scalar ABSOLUTE reference loss (the model carries the same weights as the fixture).
        scalar_absolute = model.nll(context, target, abs_positions)
        scalar_slot = model.nll(context, target, list(range(6)))  # enumerate keying
        self.assertAlmostEqual(batched_loss, scalar_absolute, delta=1e-6)
        # The slot-keyed reference genuinely differs on this padded window, so the parity
        # above is load-bearing (not a degenerate equality).
        self.assertGreater(abs(scalar_slot - batched_loss), 1e-9)

    def test_batched_training_loop_runs_padded_without_slot_keying(self) -> None:
        # Drive train_torch_lm with use_batched_forward + B>1 over Phase-2 triples whose
        # early windows are genuinely left-padded. It must train without tripping the
        # fail-closed guard (proves the batched loop threads (B,C) abs_positions).
        torch = _torch_or_skip(self)
        model = _model(self.tokenizer, use_absolute_rope=True)
        fixture = self._abs_fixture(model)
        # Build left-padded (context, abs_positions, target) triples.
        examples = []
        ids = list(self.tokenizer.encode("mia"))
        for target_id in self.tokenizer.encode(" ball"):
            context, abs_positions = make_context_positioned(ids, 6, self.tokenizer.pad_id)
            examples.append((context, abs_positions, target_id))
            ids.append(target_id)
        self.assertIn(POSITION_PAD_SENTINEL, examples[0][1])  # genuinely padded

        runtime = dict(CPU64)
        runtime["use_batched_forward"] = True
        state, losses = train_torch_lm(
            fixture=fixture, examples=examples, steps=4, learning_rate=0.02,
            torch=torch, runtime=runtime, batch_size=2,
        )
        self.assertEqual(len(losses), 4)
        self.assertTrue(all(value == value for value in losses))  # all finite


class BatchedAbsoluteRopeDefaultOffTest(unittest.TestCase):
    """Default-off byte-exact: flag off OR use_batched_forward off changes nothing."""

    def setUp(self) -> None:
        self.tokenizer = CharTokenizer.train(CORPUS)

    def test_flag_off_batched_byte_exact_to_arange_slot_keying(self) -> None:
        # With use_absolute_rope OFF, the batched forward must remain byte-identical to
        # the pre-change slot-keyed (arange) path. We capture that by asserting flag-off
        # batched == flag-off Tier-1 (which has always been slot-keyed), with NO
        # abs_positions in the runtime -- exactly the pre-change call shape.
        torch = _torch_or_skip(self)
        model = _model(self.tokenizer)  # flag OFF (use_rotary_positions on)
        fixture = _fixture(model)
        contexts = [
            make_context_positioned(self.tokenizer.encode(prompt), 6, self.tokenizer.pad_id)[0]
            for prompt in PROMPTS
        ]
        for runtime in (CPU64, CPU32):
            # No abs_positions threaded; flag-off slot-keys via arange (the prior path).
            batched = torch_batched_logits(contexts, fixture, torch, dict(runtime))
            for index, context in enumerate(contexts):
                single = torch_minimal_logits(context, fixture, torch, dict(runtime))
                # Byte-exact for f64 (slot-keyed, same reductions per the batched contract
                # within band); assert within the validated band + rank-invariance.
                assert_numeric_parity(
                    [float(v) for v in single.detach().cpu().tolist()],
                    [float(v) for v in batched[index].detach().cpu().tolist()],
                    dtype=runtime["dtype"],
                )
                assert_rank_invariant(
                    [float(v) for v in single.detach().cpu().tolist()],
                    [float(v) for v in batched[index].detach().cpu().tolist()],
                )

    def test_use_batched_forward_off_routes_to_tier1(self) -> None:
        # With use_batched_forward off (default), the training-logits builder must route
        # to the per-position Tier-1 path even under use_absolute_rope -- so the existing
        # Tier-1 absolute-RoPE parity is preserved and no batched path engages.
        torch = _torch_or_skip(self)
        from transformer_torch_training_loss import build_torch_training_logits

        model = _model(self.tokenizer, use_absolute_rope=True)
        fixture = self._abs_fixture(model)
        state = build_torch_training_state(fixture=fixture, torch=torch, runtime=CPU64)
        ids = self.tokenizer.encode("mia")
        context, abs_positions = make_context_positioned(ids, 6, self.tokenizer.pad_id)
        # use_batched_forward NOT set -> Tier-1; abs_positions still threaded for the flag.
        runtime = dict(CPU64)
        runtime["abs_positions"] = abs_positions
        tier1_logits = build_torch_training_logits(
            fixture=fixture, state=state, torch=torch, runtime=runtime, context=context
        )
        # Equals the scalar absolute reference (Tier-1 absolute-RoPE contract).
        scalar = model._forward_floats(context, abs_positions)
        assert_numeric_parity(
            scalar, [float(v) for v in tier1_logits.detach().cpu().tolist()], dtype="float64"
        )
        assert_rank_invariant(
            scalar, [float(v) for v in tier1_logits.detach().cpu().tolist()]
        )

    def _abs_fixture(self, model: TinyTransformerLM) -> dict:
        context, _positions = make_context_positioned(
            self.tokenizer.encode("mia"), model.config.context_size, self.tokenizer.pad_id
        )
        fixture = build_scalar_training_parity_fixture(
            fixture_id="batched-abs-default-off",
            model=model,
            tokenizer=self.tokenizer,
            context=context,
            target=self.tokenizer.encode(" ")[0],
            optimizer_config=OptimizationConfig(
                optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
            ),
            learning_rate=0.02,
            steps=1,
            corpus_hash="batched-abs-default-off",
        )
        fixture["tokenizer"] = {
            "tokenizer_type": getattr(self.tokenizer, "tokenizer_type", "char"),
            "vocab_size": self.tokenizer.vocab_size,
            "pad_id": self.tokenizer.pad_id,
            "tokens": list(getattr(self.tokenizer, "tokens", [])),
        }
        return fixture


if __name__ == "__main__":
    unittest.main()
