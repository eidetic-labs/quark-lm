from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_char_ops import make_context
from support.char_model import char_model_fixture, context_and_target
from support.fake_torch import fake_torch_importer
from tokenizer import CharTokenizer
from transformer_backend_parity_fixture import build_scalar_backend_parity_fixture
from transformer_model import OptimizationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_backend import (
    TORCH_RUNTIME_REPORT_KIND,
    build_torch_backend_parity_candidate,
    build_torch_training_parity_candidate,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchCandidateRuntimeReportTests(unittest.TestCase):
    def test_backend_candidate_carries_runtime_report(self) -> None:
        candidate = build_torch_backend_parity_candidate(
            fixture=_backend_fixture(),
            importer=fake_torch_importer(),
        )

        report = candidate["runtime_report"]
        self.assertEqual(report["kind"], TORCH_RUNTIME_REPORT_KIND)
        self.assertEqual(report["runtime"], candidate["runtime"])
        self.assertEqual(report["evidence_scope"], "runtime_preflight_only")
        self.assertFalse(report["parity_attempt_allowed"])
        self.assertFalse(report["training_evidence_allowed"])
        self.assertFalse(report["closed_world_boundary"]["learned_assets_imported"])

    def test_training_candidate_carries_runtime_report(self) -> None:
        candidate = build_torch_training_parity_candidate(
            fixture=_training_fixture(),
            importer=fake_torch_importer(
                training_runtime=True,
                gradient_runtime=True,
            ),
        )

        report = candidate["runtime_report"]
        self.assertEqual(report["kind"], TORCH_RUNTIME_REPORT_KIND)
        self.assertEqual(report["runtime"], candidate["runtime"])
        self.assertEqual(report["evidence_scope"], "runtime_preflight_only")
        self.assertFalse(report["parity_attempt_allowed"])
        self.assertFalse(report["training_evidence_allowed"])
        self.assertFalse(candidate["training_replay_parity_gate"]["passed"])


def _backend_fixture() -> dict:
    tokenizer = CharTokenizer.train("abc ")
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=17,
        )
    )
    context = make_context(tokenizer.encode("ab"), 4, tokenizer.pad_id)
    return build_scalar_backend_parity_fixture(
        fixture_id="tiny-scalar",
        model=model,
        tokenizer=tokenizer,
        contexts=[context],
        targets=[tokenizer.stoi["c"]],
        prompts=["ab"],
        corpus_hash="corpus-hash",
        max_new_chars=1,
    )


def _training_fixture() -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="tiny-training-scalar",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=OptimizationConfig(
            optimizer="adamw",
            gradient_accumulation_steps=2,
        ),
        learning_rate=0.02,
        steps=2,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
