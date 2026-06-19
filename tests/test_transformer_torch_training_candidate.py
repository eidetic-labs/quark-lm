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
    TORCH_TRAINING_PARITY_CANDIDATE_KIND,
    TORCH_TRAINING_REPLAY_PARITY_STATUS,
    TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
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

    def test_candidate_marks_forward_only_runtime_as_incomplete(self) -> None:
        fixture = _scalar_training_fixture()

        candidate = build_torch_training_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_training_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(
            candidate["implementation_status"],
            TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
        )
        self.assertEqual(candidate["training_readiness"]["status"], "pending")
        self.assertIn(
            "autograd",
            candidate["training_readiness"]["summary"]["failed_checks"],
        )
        self.assertEqual(candidate["training_state"]["status"], "not_built")
        self.assertEqual(candidate["initial_loss_probe"]["status"], "not_run")
        self.assertEqual(candidate["backward_probe"]["status"], "not_run")
        self.assertEqual(candidate["optimizer_step_probe"]["status"], "not_run")
        self.assertEqual(
            candidate["optimizer_step_execution_probe"]["status"],
            "not_run",
        )
        self.assertEqual(
            candidate["training_case"]["reason"],
            "pytorch training runtime is missing required capabilities",
        )
        self.assertFalse(report["passed"])

    def test_candidate_stays_pending_until_replay_parity_matches(self) -> None:
        fixture = _scalar_training_fixture()

        candidate = build_torch_training_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(
                training_runtime=True,
                gradient_runtime=True,
            ),
        )
        report = build_training_parity_report(fixture=fixture, candidate=candidate)

        self.assertTrue(candidate["runtime"]["available"])
        self.assertEqual(candidate["training_readiness"]["status"], "ready")
        self.assertEqual(candidate["training_state"]["status"], "built")
        self.assertEqual(
            candidate["training_state"]["parameter_count"],
            fixture["parameter_manifest"]["parameter_count"],
        )
        self.assertEqual(candidate["initial_loss_probe"]["status"], "matched")
        self.assertLessEqual(candidate["initial_loss_probe"]["loss_abs_diff"], 1e-9)
        self.assertEqual(candidate["backward_probe"]["status"], "gradients_available")
        self.assertEqual(
            candidate["accumulation_replay_plan"]["status"],
            "accumulation_replay_pending",
        )
        self.assertEqual(
            candidate["accumulation_replay_plan"]["microstep_count"],
            fixture["training_case"]["steps"],
        )
        self.assertFalse(
            candidate["accumulation_replay_plan"]["execution_status"][
                "replayed_backward_passes"
            ]
        )
        self.assertEqual(
            candidate["optimizer_step_contract"],
            fixture["optimizer_step_contract"],
        )
        self.assertEqual(
            candidate["optimizer_step_probe"]["status"],
            "ready_for_optimizer_execution",
        )
        self.assertEqual(
            candidate["optimizer_step_probe"]["gradient_summary"][
                "gradient_parameter_count"
            ],
            fixture["parameter_manifest"]["parameter_count"],
        )
        self.assertEqual(
            candidate["optimizer_step_execution_probe"]["status"],
            "step_control_matched",
        )
        self.assertTrue(
            candidate["optimizer_step_execution_probe"][
                "step_records_match_contract"
            ]
        )
        self.assertEqual(
            candidate["optimizer_step_execution_probe"]["gradient_clip"][
                "status"
            ],
            "gradient_clip_applied",
        )
        self.assertEqual(
            candidate["optimizer_step_execution_probe"]["gradient_accumulation"][
                "status"
            ],
            "gradient_accumulation_recorded",
        )
        self.assertTrue(
            candidate["optimizer_step_execution_probe"]["gradient_accumulation"][
                "requires_replayed_backward_passes"
            ]
        )
        self.assertFalse(
            candidate["optimizer_step_execution_probe"]["gradient_accumulation"][
                "accumulated_gradient_parity_proven"
            ]
        )
        self.assertIn(
            "clipped_gradient_buffer",
            candidate["optimizer_step_execution_probe"]["gradient_accumulation"][
                "pytorch_accumulation_readiness"
            ]["missing_requirements"],
        )
        self.assertEqual(
            candidate["optimizer_step_execution_probe"]["parameter_mutation"][
                "status"
            ],
            "parameter_mutation_not_observed",
        )
        self.assertEqual(
            candidate["optimizer_step_execution_probe"][
                "parameter_signature_comparison"
            ]["status"],
            "parameter_signature_mismatch",
        )
        self.assertEqual(
            candidate["optimizer_step_execution_probe"][
                "adamw_update_signature_comparison"
            ]["status"],
            "parameter_signature_matched",
        )
        self.assertEqual(
            candidate["implementation_status"],
            TORCH_TRAINING_REPLAY_PARITY_STATUS,
        )
        gate = candidate["training_replay_parity_gate"]
        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PARITY_STATUS)
        self.assertFalse(gate["passed"])
        self.assertIn("replay_buffer", gate["summary"]["failed_checks"])
        self.assertEqual(candidate["backend"]["parity_status"], "pending")
        self.assertEqual(candidate["optimizer_config"], fixture["optimizer_config"])
        self.assertEqual(
            candidate["parameter_manifest"],
            fixture["parameter_manifest"],
        )
        self.assertEqual(candidate["training_case"]["status"], "pending")
        self.assertEqual(
            candidate["training_case"]["reason"],
            "one or more replay parity gates have not matched scalar training evidence",
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
