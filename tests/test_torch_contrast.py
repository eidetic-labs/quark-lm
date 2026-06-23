"""Torch entity-paired contrast achieves the abstention sign-flip.

The answer-vs-unknown margin flips sign with the entity (owner prefers its
concrete answer; the entity-swapped non-owner prefers the abstain token) on the
SAME weights -- entity-conditioned abstention. This is the result the scalar
engine could not reach: without position-projection it stalled at indifference,
and it could not run position-projection in feasible time. Kept minimal (short
prompts where the entity is in-window, tiny answers) so it stays fast.
Skip-safe without torch.
"""

from __future__ import annotations

import unittest
from importlib import import_module

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_examples import AnswerExample
from neural_char_ops import make_context, make_context_positioned
from support.core import CharTokenizer, OptimizationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_contrast import (
    torch_answer_sequence_loss,
    train_torch_answer_mixed,
    train_torch_contrast,
)
from transformer_training_parity_fixture import build_scalar_training_parity_fixture

RUNTIME = {"dtype": "float64", "device": "cpu"}
PAIRS = [
    (AnswerExample("mia ball", " a", "f"), AnswerExample("noah ball", " u", "o")),
    (AnswerExample("noah cup", " b", "f"), AnswerExample("mia cup", " u", "o")),
]
ALL_TEXT = "".join(e.prompt + e.target for pair in PAIRS for e in pair) + "\n"


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class TorchContrastSignFlipTest(unittest.TestCase):
    def test_margin_flips_sign_with_entity(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=16,
            embedding_dim=8,
            feedforward_dim=16,
            use_prompt_position_projection=True,
            seed=11,
        )
        model = TinyTransformerLM.init_random(config)
        c0 = make_context(tokenizer.encode(PAIRS[0][0].prompt), 16, tokenizer.pad_id)
        fixture = build_scalar_training_parity_fixture(
            fixture_id="contrast",
            model=model,
            tokenizer=tokenizer,
            context=c0,
            target=tokenizer.encode(PAIRS[0][0].target)[0],
            optimizer_config=OptimizationConfig(
                optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
            ),
            learning_rate=0.02,
            steps=1,
            corpus_hash="x",
        )

        state, _losses = train_torch_contrast(
            fixture=fixture, tokenizer=tokenizer, pairs=PAIRS,
            steps=80, learning_rate=0.02, torch=torch, runtime=RUNTIME,
        )

        def margin(prompt: str, concrete: str) -> float:
            pid = tokenizer.encode(prompt)
            nll_unknown = float(torch_answer_sequence_loss(
                fixture=fixture, state=state, prompt_ids=pid,
                target_ids=tokenizer.encode(" u"), torch=torch, runtime=RUNTIME).detach())
            nll_concrete = float(torch_answer_sequence_loss(
                fixture=fixture, state=state, prompt_ids=pid,
                target_ids=tokenizer.encode(concrete), torch=torch, runtime=RUNTIME).detach())
            return nll_unknown - nll_concrete

        for in_example, ooc_example in PAIRS:
            self.assertGreater(margin(in_example.prompt, in_example.target), 0.0)
            self.assertLess(margin(ooc_example.prompt, in_example.target), 0.0)

    def test_mixed_objective_trains_and_stays_finite(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size, context_size=16, embedding_dim=8,
            feedforward_dim=16, seed=11,
        )
        model = TinyTransformerLM.init_random(config)
        examples = []
        for in_example, _ooc in PAIRS:
            ids = list(tokenizer.encode(in_example.prompt))
            for target_id in tokenizer.encode(in_example.target):
                context, abs_positions = make_context_positioned(ids, 16, tokenizer.pad_id)
                examples.append((context, abs_positions, target_id))
                ids.append(target_id)
        c0, _a0, t0 = examples[0]
        fixture = build_scalar_training_parity_fixture(
            fixture_id="mixed", model=model, tokenizer=tokenizer, context=c0, target=t0,
            optimizer_config=OptimizationConfig(optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0),
            learning_rate=0.02, steps=1, corpus_hash="x",
        )
        state, losses = train_torch_answer_mixed(
            fixture=fixture, tokenizer=tokenizer, examples=examples, contrast_pairs=PAIRS,
            steps=40, learning_rate=0.02, contrast_weight=1.0, torch=torch, runtime=RUNTIME,
        )
        # The joint next-token + contrast objective optimizes and never goes non-finite.
        self.assertEqual(len(losses), 40)
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(all(value == value for value in losses))
        # Update-health telemetry recorded per step (Phase 4 cross-cutting invariant).
        self.assertEqual(len(state["grad_norms"]), 40)
        self.assertTrue(all(value == value and value >= 0.0 for value in state["grad_norms"]))


def _build_mixed_fixture(torch):
    """Shared tiny mixed-objective setup (tokenizer, examples, fixture)."""

    tokenizer = CharTokenizer.train(ALL_TEXT)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, context_size=16, embedding_dim=8,
        feedforward_dim=16, seed=11,
    )
    model = TinyTransformerLM.init_random(config)
    examples = []
    for in_example, _ooc in PAIRS:
        ids = list(tokenizer.encode(in_example.prompt))
        for target_id in tokenizer.encode(in_example.target):
            context, abs_positions = make_context_positioned(ids, 16, tokenizer.pad_id)
            examples.append((context, abs_positions, target_id))
            ids.append(target_id)
    c0, _a0, t0 = examples[0]
    fixture = build_scalar_training_parity_fixture(
        fixture_id="mixed-best", model=model, tokenizer=tokenizer, context=c0, target=t0,
        optimizer_config=OptimizationConfig(
            optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
        ),
        learning_rate=0.02, steps=1, corpus_hash="x",
    )
    return tokenizer, examples, fixture


class TorchMixedCombinedBestTest(unittest.TestCase):
    def test_new_params_unset_is_byte_exact_with_baseline(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, examples, fixture = _build_mixed_fixture(torch)
        kwargs = dict(
            fixture=fixture, tokenizer=tokenizer, examples=examples, contrast_pairs=PAIRS,
            steps=30, learning_rate=0.02, contrast_weight=1.0, torch=torch, runtime=RUNTIME, seed=7,
        )
        baseline_state, baseline_losses = train_torch_answer_mixed(**kwargs)
        # Re-run with the new opt-in params explicitly UNSET (defaults) -> must match.
        tokenizer2, examples2, fixture2 = _build_mixed_fixture(torch)
        kwargs2 = dict(
            fixture=fixture2, tokenizer=tokenizer2, examples=examples2, contrast_pairs=PAIRS,
            steps=30, learning_rate=0.02, contrast_weight=1.0, torch=torch, runtime=RUNTIME, seed=7,
            validation_probe_paths=None, eval_every=0, eval_responder=None,
        )
        state, losses = train_torch_answer_mixed(**kwargs2)
        self.assertEqual(losses, baseline_losses)  # byte-exact loss trace
        base_params = [p["tensor"] for p in baseline_state["parameters"]]
        new_params = [p["tensor"] for p in state["parameters"]]
        for base, new in zip(base_params, new_params):
            self.assertTrue(bool(torch.equal(base.detach(), new.detach())))
        # No best-ckpt keys are stashed when the feature is off.
        self.assertNotIn("best_combined_score", state)

    def test_eval_every_selects_non_last_step(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, examples, fixture = _build_mixed_fixture(torch)
        # Injected eval seam: a mid-run peak (step 20) beats the last (step 30).
        reports = {
            10: {"abstention_f1": 0.90, "concrete_generation_exact_rate": 0.40},
            20: {"abstention_f1": 0.95, "concrete_generation_exact_rate": 0.80},  # peak
            30: {"abstention_f1": 0.99, "concrete_generation_exact_rate": 0.10},
        }

        def fake_eval(step):
            head = reports[step]
            return {
                "headline": head,
                "provenance": {"candidate_menus": "per_type"},
            }

        state, _losses = train_torch_answer_mixed(
            fixture=fixture, tokenizer=tokenizer, examples=examples, contrast_pairs=PAIRS,
            steps=30, learning_rate=0.02, contrast_weight=1.0, torch=torch, runtime=RUNTIME, seed=7,
            validation_probe_paths=["unused"], eval_every=10, eval_responder=object(),
            f1_floor=0.85, gen_floor=0.05, _eval_report_fn=fake_eval,
        )
        self.assertEqual(state["best_step"], 20)  # the mid-run peak, not the last step
        self.assertAlmostEqual(state["best_abstention_f1"], 0.95)
        self.assertAlmostEqual(state["best_concrete_gen"], 0.80)
        self.assertGreater(state["best_combined_score"], 0.0)

    def test_no_step_clears_floors_keeps_last_and_reports(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, examples, fixture = _build_mixed_fixture(torch)
        # Every eval is an over-abstainer: F1 high, concrete-gen below floor -> 0 score.
        def fake_eval(step):
            return {
                "headline": {"abstention_f1": 0.98, "concrete_generation_exact_rate": 0.0},
                "provenance": {"candidate_menus": "per_type"},
            }

        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            state, _losses = train_torch_answer_mixed(
                fixture=fixture, tokenizer=tokenizer, examples=examples, contrast_pairs=PAIRS,
                steps=20, learning_rate=0.02, contrast_weight=1.0, torch=torch, runtime=RUNTIME, seed=7,
                validation_probe_paths=["unused"], eval_every=10, eval_responder=object(),
                f1_floor=0.85, gen_floor=0.05, _eval_report_fn=fake_eval,
            )
        self.assertIsNone(state["best_combined_score"])
        self.assertIn("no does-both checkpoint found", buffer.getvalue())

    def test_global_pool_provenance_raises(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer, examples, fixture = _build_mixed_fixture(torch)

        def fake_eval(step):
            return {
                "headline": {"abstention_f1": 0.9, "concrete_generation_exact_rate": 0.5},
                "provenance": {"candidate_menus": "global_pool"},  # contaminated
            }

        with self.assertRaises(ValueError):
            train_torch_answer_mixed(
                fixture=fixture, tokenizer=tokenizer, examples=examples, contrast_pairs=PAIRS,
                steps=10, learning_rate=0.02, contrast_weight=1.0, torch=torch, runtime=RUNTIME, seed=7,
                validation_probe_paths=["unused"], eval_every=10, eval_responder=object(),
                _eval_report_fn=fake_eval,
            )


if __name__ == "__main__":
    unittest.main()
