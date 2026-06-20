from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION,
    TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
    build_torch_training_backend_promotion_gate,
)
from transformer_torch_training_promotion_gate_validation import (
    validate_torch_training_backend_promotion_gate,
)


class TransformerTorchTrainingPromotionGateTests(unittest.TestCase):
    def test_matched_replay_parity_remains_unpromoted_fixture_evidence(self) -> None:
        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=_boundary(),
        )

        self.assertEqual(gate["status"], TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS)
        self.assertEqual(
            gate["schema_version"],
            TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["promotion_eligible"])
        self.assertFalse(gate["promoted_training_backend"])
        self.assertTrue(gate["parity_evidence_matched"])
        self.assertTrue(gate["closed_world_boundary_passed"])
        self.assertEqual(gate["closed_world_boundary_failures"], [])
        self.assertEqual(
            [check["name"] for check in gate["checks"]],
            list(TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS),
        )
        self.assertEqual(
            gate["blockers"],
            ["fixture_scope_only", "model_quality_gate"],
        )
        self.assertEqual(
            gate["required_future_gates"],
            list(TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES),
        )
        validate_torch_training_backend_promotion_gate(
            gate,
            closed_world_boundary=_boundary(),
        )

    def test_failed_training_parity_blocks_before_future_promotion_gates(self) -> None:
        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="pending", replay_passed=False),
            report={"passed": False},
            closed_world_boundary=_boundary(),
        )

        self.assertFalse(gate["parity_evidence_matched"])
        self.assertEqual(
            gate["blockers"],
            [
                "training_parity_report",
                "fixture_scope_only",
                "model_quality_gate",
            ],
        )

    def test_stale_replay_status_does_not_match_parity_evidence(self) -> None:
        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(
                parity_status="matched",
                replay_passed=True,
                replay_status="training_replay_parity_pending",
            ),
            report={"passed": True},
            closed_world_boundary=_boundary(),
        )

        self.assertFalse(gate["parity_evidence_matched"])
        self.assertEqual(
            gate["blockers"],
            ["fixture_scope_only", "model_quality_gate"],
        )

    def test_closed_world_boundary_violation_blocks_promotion(self) -> None:
        boundary = _boundary()
        boundary["pretrained_weights_imported"] = True

        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=boundary,
        )

        self.assertFalse(gate["closed_world_boundary_passed"])
        self.assertEqual(
            gate["closed_world_boundary_failures"],
            ["pretrained_weights_imported"],
        )
        self.assertIn("closed_world_boundary", gate["blockers"])
        validate_torch_training_backend_promotion_gate(
            gate,
            closed_world_boundary=boundary,
        )

    def test_closed_world_boundary_requires_runtime_library_allowance(self) -> None:
        boundary = _boundary()
        boundary["runtime_library_allowed"] = False

        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=boundary,
        )

        self.assertFalse(gate["closed_world_boundary_passed"])
        self.assertEqual(
            gate["closed_world_boundary_failures"],
            ["runtime_library_allowed"],
        )
        self.assertIn("closed_world_boundary", gate["blockers"])

    def test_closed_world_boundary_requires_admitted_curriculum_source(self) -> None:
        boundary = _boundary()
        boundary["training_text_source"] = "external_corpus"

        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=boundary,
        )

        self.assertFalse(gate["closed_world_boundary_passed"])
        self.assertEqual(
            gate["closed_world_boundary_failures"],
            ["training_text_source"],
        )
        self.assertIn("closed_world_boundary", gate["blockers"])

    def test_closed_world_boundary_reports_multiple_failures(self) -> None:
        boundary = _boundary()
        boundary["runtime_library_allowed"] = False
        boundary["training_text_source"] = "external_corpus"
        boundary["external_embeddings_imported"] = True

        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=boundary,
        )

        self.assertEqual(
            gate["closed_world_boundary_failures"],
            [
                "runtime_library_allowed",
                "training_text_source",
                "external_embeddings_imported",
            ],
        )

    def test_validator_rejects_stale_promotion_gate_blockers(self) -> None:
        gate = _gate()
        gate["blockers"] = ["fixture_scope_only"]

        with self.assertRaisesRegex(ValueError, "blockers"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )

    def test_validator_rejects_stale_promotion_gate_future_gates(self) -> None:
        gate = _gate()
        gate["required_future_gates"] = ["general_training_backend_gate"]

        with self.assertRaisesRegex(ValueError, "future gates"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )

    def test_validator_rejects_stale_promotion_gate_boundary_state(self) -> None:
        boundary = _boundary()
        boundary["pretrained_weights_imported"] = True
        gate = _gate()

        with self.assertRaisesRegex(ValueError, "boundary status"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=boundary,
            )

    def test_validator_rejects_non_boolean_parity_status(self) -> None:
        gate = _gate()
        gate["parity_evidence_matched"] = "false"

        with self.assertRaisesRegex(ValueError, "parity status"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )


def _candidate(
    *,
    parity_status: str,
    replay_passed: bool,
    replay_status: str = "training_replay_parity_matched",
) -> dict:
    return {
        "backend": {"parity_status": parity_status},
        "training_replay_parity_gate": {
            "status": replay_status,
            "passed": replay_passed,
            "parity_status": parity_status,
        },
    }


def _gate() -> dict:
    return build_torch_training_backend_promotion_gate(
        candidate=_candidate(parity_status="matched", replay_passed=True),
        report={"passed": True},
        closed_world_boundary=_boundary(),
    )


def _boundary() -> dict:
    return {
        "runtime_library_allowed": True,
        "training_text_source": "admitted_curriculum",
        "learned_assets_imported": False,
        "training_data_imported": False,
        "pretrained_weights_imported": False,
        "pretrained_tokenizer_imported": False,
        "external_embeddings_imported": False,
    }
