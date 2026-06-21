"""Phase 1 parity contract: torch ranks candidates exactly as the scalar reference.

Pure-logic checks for the ranking/tie-break/tolerance helpers, plus an end-to-end
rank-invariance check that the torch forward (at float64 AND float32) ranks a set
of answer candidates identically to the scalar reference on the SAME weights.
Skip-safe when torch is not installed.
"""

from __future__ import annotations

import unittest
from importlib import import_module

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_examples import AnswerExample
from neural_char_ops import make_context
from support.core import CharTokenizer, OptimizationConfig, TransformerConfig
from transformer_direct_answer_core import answer_sequence_nll
from transformer_parity_contract import (
    assert_rank_invariant,
    candidate_ranking,
    numeric_parity_violations,
    validate_candidate_parity,
)
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_contrast import torch_answer_sequence_loss
from transformer_torch_training_state import build_torch_training_state
from transformer_training_parity_fixture import build_scalar_training_parity_fixture

PROBES = [
    ("mia ball", [" box", " cup", " red", " unknown"]),
    ("noah cup", [" red", " box", " unknown", " cup"]),
    ("ava book", [" unknown", " shelf", " box", " red"]),
    ("leo", [" box", " unknown", " cup"]),
    ("mia", [" red", " cup", " box", " unknown"]),
    ("the red ball is", [" box", " here", " unknown", " gone"]),
    ("noah", [" cup", " box", " unknown", " red"]),
    ("ball box red cup", [" unknown", " mia", " noah"]),
    ("ava ball cup", [" box", " unknown", " red", " shelf"]),
    ("leo book shelf", [" red", " box", " unknown", " here"]),
]
ALL_TEXT = "".join(p + "".join(c) for p, c in PROBES) + "\n"


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class ParityContractLogicTest(unittest.TestCase):
    def test_ranking_tie_break_is_by_index(self) -> None:
        # equal NLLs must resolve deterministically to ascending index order
        self.assertEqual(candidate_ranking([0.5, 0.1, 0.5, 0.1]), [1, 3, 0, 2])

    def test_numeric_violations_respect_dtype_band(self) -> None:
        self.assertEqual(numeric_parity_violations([1.0], [1.0 + 1e-7], dtype="float64"), [])
        self.assertTrue(numeric_parity_violations([1.0], [1.0 + 1e-2], dtype="float64"))

    def test_rank_invariance_flags_a_flip(self) -> None:
        with self.assertRaises(AssertionError):
            assert_rank_invariant([0.1, 0.2], [0.2, 0.1])


class TorchRankInvarianceTest(unittest.TestCase):
    def test_torch_ranks_candidates_like_scalar_at_f64_and_f32(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size, context_size=16, embedding_dim=8,
            feedforward_dim=16, use_prompt_position_projection=True, seed=11,
        )
        model = TinyTransformerLM.init_random(config)
        first = AnswerExample(PROBES[0][0], PROBES[0][1][0], "x")
        fixture = build_scalar_training_parity_fixture(
            fixture_id="parity-contract", model=model, tokenizer=tokenizer,
            context=make_context(tokenizer.encode(first.prompt), 16, tokenizer.pad_id),
            target=tokenizer.encode(first.target)[0],
            optimizer_config=OptimizationConfig(optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0),
            learning_rate=0.01, steps=1, corpus_hash="x",
        )
        # Reconstruct scalar + torch on the SAME from-scratch init weights.
        scalar_model, _ = TinyTransformerLM.from_dict(
            {"config": fixture["model_config"], "weights": fixture["initial_weights"]}
        )

        def scalar_nlls(prompt, candidates):
            return [answer_sequence_nll(scalar_model, tokenizer, AnswerExample(prompt, c, "x")) for c in candidates]

        def torch_nlls(prompt, candidates, dtype):
            runtime = {"dtype": dtype, "device": "cpu"}
            state = build_torch_training_state(fixture=fixture, torch=torch, runtime=runtime)
            pid = tokenizer.encode(prompt)
            return [
                float(torch_answer_sequence_loss(
                    fixture=fixture, state=state, prompt_ids=pid,
                    target_ids=tokenizer.encode(c), torch=torch, runtime=runtime).detach())
                for c in candidates
            ]

        scalar = [scalar_nlls(p, c) for p, c in PROBES]
        for dtype in ("float64", "float32"):
            torch_values = [torch_nlls(p, c, dtype) for p, c in PROBES]
            validate_candidate_parity(scalar, torch_values, dtype=dtype)


if __name__ == "__main__":
    unittest.main()
