from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_backend_policy import transformer_backend_metadata
from transformer_constraints import transformer_constraint_report
from transformer_direct_answer_update_outcome_constraints import (
    UPDATE_OUTCOME_CONSTRAINT_NAME,
    direct_answer_update_outcome_check,
)


class TransformerDirectAnswerUpdateOutcomeConstraintsTest(unittest.TestCase):
    def test_accepted_outcome_passes_constraint(self) -> None:
        check = direct_answer_update_outcome_check(
            {"direct_answer_weight_update_outcome": _outcome("accepted", True)}
        )

        self.assertTrue(check["passed"])

    def test_not_run_outcome_passes_constraint(self) -> None:
        check = direct_answer_update_outcome_check(
            {"direct_answer_weight_update_outcome": _outcome("not_run", True)}
        )

        self.assertTrue(check["passed"])

    def test_missing_outcome_fails_constraint(self) -> None:
        check = direct_answer_update_outcome_check({})

        self.assertFalse(check["passed"])
        self.assertEqual(
            check["details"]["error"],
            "direct-answer update outcome is missing",
        )

    def test_rejected_outcome_fails_constraint(self) -> None:
        check = direct_answer_update_outcome_check(
            {
                "direct_answer_weight_update_outcome": _outcome(
                    "rejected_frontier_restore",
                    False,
                )
            }
        )

        self.assertFalse(check["passed"])
        self.assertEqual(check["details"]["status"], "rejected_frontier_restore")
        self.assertFalse(check["details"]["accepted"])

    def test_constraint_report_blocks_rejected_update_before_quality(self) -> None:
        report = transformer_constraint_report(
            _metrics(_outcome("rejected_frontier_restore", False))
        )

        self.assertEqual(report["status"], "blocked_before_quality_metrics")
        self.assertIn(UPDATE_OUTCOME_CONSTRAINT_NAME, report["failed_constraints"])
        self.assertFalse(report["quality_metrics_considered"])


def _outcome(status: str, accepted: bool) -> dict[str, Any]:
    return {
        "status": status,
        "accepted": accepted,
        "reason": "test",
        "direct_steps_to_run": 1 if status != "not_run" else 0,
        "restored_best_branch_snapshot": False,
        "restored_frontier_progress_snapshot": not accepted,
    }


def _metrics(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "run-001",
        "baseline": {"step": 0},
        "final": {"step": 1},
        "training_data": "answer_model corpus-derived AnswerExample lessons",
        "closed_world_verifier": {"passed": True},
        "sweep_plan": {"kind": "transformer_sweep_plan"},
        "replay_mixture_report": {"summary": {"passed": True}},
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "backend": transformer_backend_metadata(seed=17, tokenizer_type="char"),
        "direct_answer": {
            "direct_answer_weight_update_outcome": outcome,
            "direct_answer_branch_context_gate": {"passed": True},
            "baseline": {"branch_target_coverage_by_profile": {"qa": 1.0}},
            "final": {
                "branch_diversity_target": {"passed": True},
                "branch_target_coverage_by_profile": {"qa": 1.0},
                "evals": {"qa": {"count": 1, "exact": 1}},
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
