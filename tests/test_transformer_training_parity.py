from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from transformer_backend_policy import PYTORCH_BACKEND, transformer_backend_metadata
from transformer_model import OptimizationConfig
from transformer_training_parity import (
    TRAINING_PARITY_FIXTURE_KIND,
    build_scalar_training_parity_fixture,
    build_training_parity_report,
    validate_training_parity_fixture,
)


class TransformerTrainingParityTests(unittest.TestCase):
    def test_scalar_training_fixture_records_optimizer_and_training_evidence(self) -> None:
        fixture = _scalar_training_fixture()

        validate_training_parity_fixture(fixture)
        case = fixture["training_case"]
        self.assertEqual(fixture["kind"], TRAINING_PARITY_FIXTURE_KIND)
        self.assertEqual(fixture["reference_backend"]["backend"], "scalar_python")
        self.assertEqual(fixture["optimizer_config"]["optimizer"], "adamw")
        self.assertEqual(len(case["step_records"]), 2)
        self.assertEqual(case["step_records"][0]["optimizer_summary"]["update_count"], 0)
        self.assertFalse(
            case["step_records"][0]["optimizer_gradient_evidence"][
                "update_applied"
            ]
        )
        self.assertFalse(
            case["step_records"][0]["optimizer_gradient_evidence"][
                "accumulated_gradient"
            ]["available"]
        )
        self.assertTrue(
            case["step_records"][1]["optimizer_gradient_evidence"][
                "update_applied"
            ]
        )
        self.assertTrue(
            case["step_records"][1]["optimizer_gradient_evidence"][
                "accumulated_gradient"
            ]["available"]
        )
        self.assertEqual(case["optimizer_state"]["update_count"], 1)
        self.assertEqual(case["optimizer_state"]["pending_accumulation"], 0)
        self.assertEqual(
            fixture["parameter_manifest"]["parameter_count"],
            case["optimizer_state"]["param_count"],
        )
        self.assertEqual(
            fixture["optimizer_step_contract"]["expected_final_optimizer_state"][
                "update_count"
            ],
            case["optimizer_state"]["update_count"],
        )
        self.assertGreater(case["initial_loss"], case["final_loss"])
        self.assertGreater(case["parameter_signature"]["count"], 0)
        self.assertEqual(
            case["trainable_parameter_signature"]["count"],
            fixture["parameter_manifest"]["parameter_count"],
        )
        self.assertLess(
            case["trainable_parameter_signature"]["count"],
            case["parameter_signature"]["count"],
        )

    def test_training_report_passes_for_matching_candidate(self) -> None:
        fixture = _scalar_training_fixture()
        candidate = _matching_candidate(fixture)

        report = build_training_parity_report(fixture=fixture, candidate=candidate)

        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate_backend"], PYTORCH_BACKEND)
        self.assertEqual(report["summary"]["failed_checks"], [])

    def test_training_report_fails_when_candidate_training_drifts(self) -> None:
        fixture = _scalar_training_fixture()
        candidate = _matching_candidate(fixture)
        candidate["training_case"]["final_logits"][0] += 0.01
        candidate["training_case"]["optimizer_state"]["update_count"] = 7

        report = build_training_parity_report(fixture=fixture, candidate=candidate)

        failed = set(report["summary"]["failed_checks"])
        self.assertFalse(report["passed"])
        self.assertIn("training_final_logits", failed)
        self.assertIn("training_optimizer_state", failed)


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


def _matching_candidate(fixture: dict) -> dict:
    return {
        "backend": transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            seed=fixture["reference_backend"]["seed"],
            tokenizer_type=fixture["tokenizer"]["tokenizer_type"],
            corpus_hash=fixture["reference_backend"]["corpus_hash"],
            device="cpu",
            dtype="float32",
            parity_status="matched",
        ),
        "parameter_manifest": copy.deepcopy(fixture["parameter_manifest"]),
        "optimizer_step_contract": copy.deepcopy(
            fixture["optimizer_step_contract"]
        ),
        "training_case": copy.deepcopy(fixture["training_case"]),
    }


if __name__ == "__main__":
    unittest.main()
