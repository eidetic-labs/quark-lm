from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
    build_torch_training_backend_promotion_gate,
)


class TransformerTorchTrainingPromotionGateTests(unittest.TestCase):
    def test_matched_replay_parity_remains_unpromoted_fixture_evidence(self) -> None:
        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=_boundary(),
        )

        self.assertEqual(gate["status"], TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["promotion_eligible"])
        self.assertFalse(gate["promoted_training_backend"])
        self.assertTrue(gate["parity_evidence_matched"])
        self.assertTrue(gate["closed_world_boundary_passed"])
        self.assertEqual(
            gate["blockers"],
            ["fixture_scope_only", "model_quality_gate"],
        )
        self.assertIn("general_training_backend_gate", gate["required_future_gates"])

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

    def test_closed_world_boundary_violation_blocks_promotion(self) -> None:
        boundary = _boundary()
        boundary["pretrained_weights_imported"] = True

        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=boundary,
        )

        self.assertFalse(gate["closed_world_boundary_passed"])
        self.assertIn("closed_world_boundary", gate["blockers"])

    def test_closed_world_boundary_requires_runtime_library_allowance(self) -> None:
        boundary = _boundary()
        boundary["runtime_library_allowed"] = False

        gate = build_torch_training_backend_promotion_gate(
            candidate=_candidate(parity_status="matched", replay_passed=True),
            report={"passed": True},
            closed_world_boundary=boundary,
        )

        self.assertFalse(gate["closed_world_boundary_passed"])
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
        self.assertIn("closed_world_boundary", gate["blockers"])


def _candidate(*, parity_status: str, replay_passed: bool) -> dict:
    return {
        "backend": {"parity_status": parity_status},
        "training_replay_parity_gate": {
            "passed": replay_passed,
            "parity_status": parity_status,
        },
    }


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
