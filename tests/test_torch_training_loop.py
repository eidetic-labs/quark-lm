"""Phase 3a: the torch training loop trains and matches the scalar reference.

Parity: with gradient_accumulation_steps=1 and no schedule, training a torch
model from a from-scratch fixture's initial weights reproduces the scalar
reference's final loss (the fixture trains a scalar model internally and records
final_loss). Skip-safe when torch is not installed.
"""

from __future__ import annotations

import unittest
from importlib import import_module

import support  # noqa: F401  (inserts src/ onto sys.path)
from support.char_model import char_model_fixture, context_and_target
from support.core import OptimizationConfig
from transformer_training_parity_fixture import build_scalar_training_parity_fixture
from transformer_torch_training_loop import (
    eval_torch_loss,
    save_torch_checkpoint,
    train_torch_lm,
)
from transformer_torch_training_state import build_torch_training_state

RUNTIME = {"dtype": "float64", "device": "cpu"}


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


def _fixture(model, tokenizer, context, target, *, steps, learning_rate):
    return build_scalar_training_parity_fixture(
        fixture_id="loop-parity",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=OptimizationConfig(
            optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
        ),
        learning_rate=learning_rate,
        steps=steps,
        corpus_hash="loop-test",
    )


class TorchTrainingLoopTest(unittest.TestCase):
    def test_torch_loop_matches_scalar_final_loss(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
        context, target = context_and_target(ids, config, tokenizer)
        fixture = _fixture(model, tokenizer, context, target, steps=5, learning_rate=0.05)

        state, _losses = train_torch_lm(
            fixture=fixture,
            examples=[(context, target)],
            steps=5,
            learning_rate=0.05,
            torch=torch,
            runtime=RUNTIME,
        )
        torch_final = eval_torch_loss(
            fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=RUNTIME
        )

        # Torch training reproduces the scalar reference (and actually learned).
        self.assertAlmostEqual(torch_final, fixture["training_case"]["final_loss"], delta=1e-6)
        self.assertLess(torch_final, fixture["training_case"]["initial_loss"])

    def test_torch_loop_trains_multiple_examples(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abcabc\n", seed=7)
        examples = [context_and_target(ids, config, tokenizer, index=i) for i in (2, 3, 4)]
        ctx0, tgt0 = examples[0]
        fixture = _fixture(model, tokenizer, ctx0, tgt0, steps=2, learning_rate=0.05)

        untrained = build_torch_training_state(fixture=fixture, torch=torch, runtime=RUNTIME)
        before = sum(
            eval_torch_loss(fixture=fixture, state=untrained, context=c, target=t, torch=torch, runtime=RUNTIME)
            for c, t in examples
        ) / len(examples)

        state, _losses = train_torch_lm(
            fixture=fixture, examples=examples, steps=30, learning_rate=0.05, torch=torch, runtime=RUNTIME
        )
        after = sum(
            eval_torch_loss(fixture=fixture, state=state, context=c, target=t, torch=torch, runtime=RUNTIME)
            for c, t in examples
        ) / len(examples)

        self.assertLess(after, before)

    def test_torch_checkpoint_round_trips_to_scalar_model(self) -> None:
        import tempfile
        from pathlib import Path

        from transformer_tiny_lm import TinyTransformerLM

        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
        context, target = context_and_target(ids, config, tokenizer)
        fixture = _fixture(model, tokenizer, context, target, steps=5, learning_rate=0.05)

        state, _losses = train_torch_lm(
            fixture=fixture,
            examples=[(context, target)],
            steps=5,
            learning_rate=0.05,
            torch=torch,
            runtime=RUNTIME,
        )
        torch_loss = eval_torch_loss(
            fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=RUNTIME
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "torch_answer.json"
            save_torch_checkpoint(path, fixture=fixture, state=state, tokenizer=tokenizer)
            reloaded, reloaded_tokenizer = TinyTransformerLM.load(path)

        # The torch-trained checkpoint loads as a normal model and the scalar
        # forward scores it identically to the torch state -> the spine can read it.
        self.assertIsNotNone(reloaded_tokenizer)
        self.assertAlmostEqual(reloaded.nll(context, target), torch_loss, delta=1e-6)

    def test_lr_warmup_changes_torch_training(self) -> None:
        torch = _torch_or_skip(self)

        def final_loss(warmup_steps: int) -> float:
            tokenizer, ids, config, model = char_model_fixture("abcabc\n", seed=7)
            examples = [context_and_target(ids, config, tokenizer, index=i) for i in (2, 3, 4)]
            ctx0, tgt0 = examples[0]
            fixture = build_scalar_training_parity_fixture(
                fixture_id="warmup", model=model, tokenizer=tokenizer, context=ctx0, target=tgt0,
                optimizer_config=OptimizationConfig(
                    optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0,
                    warmup_steps=warmup_steps,
                ),
                learning_rate=0.1, steps=8, corpus_hash="x",
            )
            _state, losses = train_torch_lm(
                fixture=fixture, examples=examples, steps=8, learning_rate=0.1,
                torch=torch, runtime=RUNTIME,
            )
            return losses[-1]

        # Warmup ramps the LR from ~0, so the trajectory differs from constant LR.
        self.assertGreater(abs(final_loss(0) - final_loss(6)), 1e-4)

    def test_grad_norms_recorded_and_finite(self) -> None:
        import math

        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abcabc\n", seed=7)
        examples = [context_and_target(ids, config, tokenizer, index=i) for i in (2, 3, 4)]
        ctx0, tgt0 = examples[0]
        fixture = _fixture(model, tokenizer, ctx0, tgt0, steps=6, learning_rate=0.05)

        state, _losses = train_torch_lm(
            fixture=fixture, examples=examples, steps=6, learning_rate=0.05,
            torch=torch, runtime=RUNTIME,
        )

        norms = state["grad_norms"]
        # One pre-clip gradient norm per update, all finite and non-negative; a
        # learning model has at least one nonzero gradient (dead-gradient guard).
        self.assertEqual(len(norms), 6)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in norms))
        self.assertGreater(max(norms), 0.0)

    def test_per_epoch_reshuffle_is_deterministic_and_changes_trajectory(self) -> None:
        torch = _torch_or_skip(self)

        def final_loss(*, shuffle: bool) -> float:
            tokenizer, ids, config, model = char_model_fixture("abcabc\n", seed=7)
            examples = [context_and_target(ids, config, tokenizer, index=i) for i in (2, 3, 4)]
            ctx0, tgt0 = examples[0]
            fixture = _fixture(model, tokenizer, ctx0, tgt0, steps=12, learning_rate=0.05)
            _state, losses = train_torch_lm(
                fixture=fixture, examples=examples, steps=12, learning_rate=0.05,
                torch=torch, runtime=RUNTIME, seed=5, shuffle_each_epoch=shuffle,
            )
            return losses[-1]

        # Deterministic: same seed -> same per-epoch permutations -> identical result.
        self.assertEqual(final_loss(shuffle=True), final_loss(shuffle=True))
        # Reshuffling the order across epochs changes the AdamW trajectory, so the
        # final loss differs from the fixed cyclic order (the reshuffle has effect).
        self.assertNotAlmostEqual(final_loss(shuffle=True), final_loss(shuffle=False), places=9)

    def test_grad_accumulation_matches_scalar_and_counts_updates(self) -> None:
        torch = _torch_or_skip(self)

        def run(*, steps: int, accum: int):
            tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
            context, target = context_and_target(ids, config, tokenizer)
            fixture = build_scalar_training_parity_fixture(
                fixture_id="accum", model=model, tokenizer=tokenizer, context=context, target=target,
                optimizer_config=OptimizationConfig(
                    optimizer="adamw", gradient_accumulation_steps=accum, weight_decay=0.0,
                ),
                learning_rate=0.05, steps=steps, corpus_hash="x",
            )
            state, _losses = train_torch_lm(
                fixture=fixture, examples=[(context, target)], steps=steps,
                learning_rate=0.05, torch=torch, runtime=RUNTIME,
            )
            final = eval_torch_loss(
                fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=RUNTIME,
            )
            return final, state["applied_updates"], fixture["training_case"]["final_loss"]

        # N=2 over 6 micro-steps -> 3 applied updates, mean-reduced, matching the
        # scalar (which accumulates N=2 and keys LR on the applied-update count).
        final, updates, scalar_final = run(steps=6, accum=2)
        self.assertEqual(updates, 3)
        self.assertAlmostEqual(final, scalar_final, delta=1e-6)
        # 5 micro-steps, N=2 -> 2 applied updates; the trailing partial window is
        # left unapplied, exactly as the scalar never flushes a partial.
        final5, updates5, scalar_final5 = run(steps=5, accum=2)
        self.assertEqual(updates5, 2)
        self.assertAlmostEqual(final5, scalar_final5, delta=1e-6)

    def test_grad_accumulation_with_uniform_weight_decay_matches_scalar(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
        context, target = context_and_target(ids, config, tokenizer)
        fixture = build_scalar_training_parity_fixture(
            fixture_id="accum-wd", model=model, tokenizer=tokenizer, context=context, target=target,
            optimizer_config=OptimizationConfig(
                optimizer="adamw", gradient_accumulation_steps=2, weight_decay=0.01,
            ),
            learning_rate=0.05, steps=6, corpus_hash="x",
        )
        state, _losses = train_torch_lm(
            fixture=fixture, examples=[(context, target)], steps=6,
            learning_rate=0.05, torch=torch, runtime=RUNTIME,
        )
        final = eval_torch_loss(
            fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=RUNTIME,
        )
        # Uniform (single-group) weight decay composes with accumulation at parity.
        self.assertAlmostEqual(final, fixture["training_case"]["final_loss"], delta=1e-6)

    def test_grad_accumulation_lr_keyed_on_applied_updates(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
        context, target = context_and_target(ids, config, tokenizer)
        fixture = build_scalar_training_parity_fixture(
            fixture_id="accum-warmup", model=model, tokenizer=tokenizer, context=context, target=target,
            optimizer_config=OptimizationConfig(
                optimizer="adamw", gradient_accumulation_steps=2, weight_decay=0.0, warmup_steps=3,
            ),
            learning_rate=0.1, steps=8, corpus_hash="x",
        )
        state, _losses = train_torch_lm(
            fixture=fixture, examples=[(context, target)], steps=8,
            learning_rate=0.1, torch=torch, runtime=RUNTIME,
        )
        final = eval_torch_loss(
            fixture=fixture, state=state, context=context, target=target, torch=torch, runtime=RUNTIME,
        )
        # Warmup ramps over the first 3 APPLIED updates (4 total for 8 micro-steps
        # at N=2). Parity holds only if the torch loop keys the schedule on the
        # applied-update count, not the micro-step -- a micro-step keying breaks
        # this match (the discriminating test for the LR-keying contract).
        self.assertEqual(state["applied_updates"], 4)
        self.assertAlmostEqual(final, fixture["training_case"]["final_loss"], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
