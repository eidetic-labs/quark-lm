from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from support.fake_torch import fake_torch_importer
from transformer_model import OptimizationConfig
from transformer_torch_backend import (
    TORCH_TRAINING_IMPLEMENTATION_STATUS,
    TORCH_TRAINING_PARITY_CANDIDATE_KIND,
    build_torch_training_parity_candidate,
)
from transformer_training_parity import (
    build_scalar_training_parity_fixture,
    build_training_parity_report,
)


class TransformerTorchTrainingCandidateTests(unittest.TestCase):
    def test_candidate_marks_missing_runtime_as_failed(self) -> None:
        fixture = _scalar_training_fixture()

        candidate = build_torch_training_parity_candidate(
            fixture=fixture,
            importer=_missing_importer,
        )
        report = build_training_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["kind"], TORCH_TRAINING_PARITY_CANDIDATE_KIND)
        self.assertEqual(candidate["implementation_status"], "runtime_unavailable")
        self.assertFalse(candidate["runtime"]["available"])
        self.assertEqual(candidate["backend"]["parity_status"], "failed")
        self.assertEqual(candidate["training_case"]["status"], "blocked")
        self.assertFalse(report["passed"])
        self.assertNotIn("backend_metadata", report["summary"]["failed_checks"])

    def test_candidate_stays_pending_when_training_is_not_implemented(self) -> None:
        fixture = _scalar_training_fixture()

        candidate = build_torch_training_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_training_parity_report(fixture=fixture, candidate=candidate)

        self.assertTrue(candidate["runtime"]["available"])
        self.assertEqual(
            candidate["implementation_status"],
            TORCH_TRAINING_IMPLEMENTATION_STATUS,
        )
        self.assertEqual(candidate["backend"]["parity_status"], "pending")
        self.assertEqual(candidate["optimizer_config"], fixture["optimizer_config"])
        self.assertEqual(
            candidate["parameter_manifest"],
            fixture["parameter_manifest"],
        )
        self.assertEqual(candidate["training_case"]["status"], "pending")
        self.assertEqual(
            candidate["training_case"]["reason"],
            "pytorch training parity is not implemented yet",
        )
        self.assertFalse(report["passed"])
        self.assertNotIn("backend_metadata", report["summary"]["failed_checks"])
        self.assertIn("training_final_loss", report["summary"]["failed_checks"])
        self.assertIn("training_optimizer_state", report["summary"]["failed_checks"])

    def test_candidate_marks_unavailable_dtype_as_pending(self) -> None:
        fixture = _scalar_training_fixture()

        candidate = build_torch_training_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
            requested_dtype="bfloat16",
        )
        report = build_training_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["implementation_status"], "dtype_unavailable")
        self.assertEqual(candidate["backend"]["parity_status"], "pending")
        self.assertFalse(candidate["runtime"]["dtype_available"])
        self.assertEqual(candidate["training_case"]["status"], "pending")
        self.assertEqual(
            candidate["training_case"]["reason"],
            "requested pytorch dtype is unavailable",
        )
        self.assertFalse(report["passed"])


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


def _scalar_training_fixture() -> dict:
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
            warmup_steps=2,
            decay_steps=2,
            min_learning_rate=0.001,
        ),
        learning_rate=0.02,
        steps=2,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
