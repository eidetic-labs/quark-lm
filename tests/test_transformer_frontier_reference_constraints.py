from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_backend_policy import transformer_backend_metadata
from transformer_constraints import transformer_constraint_report


class TransformerFrontierReferenceConstraintsTest(unittest.TestCase):
    def test_active_reference_regression_blocks_before_quality(self) -> None:
        report = transformer_constraint_report(
            _metrics(
                {
                    "active": True,
                    "used_for_training": False,
                    "metrics_path": "runs/frontier/transformer_answer_metrics.json",
                    "frontier_run_id": "frontier",
                    "final_comparison": {
                        "available": True,
                        "passed": False,
                        "coverage_preserved": False,
                        "stability_preserved": False,
                        "score_preserved": True,
                    },
                }
            )
        )

        self.assertEqual(report["status"], "blocked_before_quality_metrics")
        self.assertIn(
            "direct_answer_frontier_reference_final",
            report["failed_constraints"],
        )
        self.assertFalse(report["quality_metrics_considered"])

    def test_active_reference_preservation_allows_quality_checks(self) -> None:
        report = transformer_constraint_report(
            _metrics(
                {
                    "active": True,
                    "used_for_training": False,
                    "metrics_path": "runs/frontier/transformer_answer_metrics.json",
                    "frontier_run_id": "frontier",
                    "final_comparison": {
                        "available": True,
                        "passed": True,
                        "coverage_preserved": True,
                        "stability_preserved": True,
                        "score_preserved": True,
                    },
                }
            )
        )

        self.assertTrue(report["constraints_passed"])
        self.assertTrue(report["quality_metrics_considered"])
        self.assertNotIn(
            "direct_answer_frontier_reference_final",
            report["failed_constraints"],
        )

    def test_frontier_training_use_blocks_promotion(self) -> None:
        report = transformer_constraint_report(
            _metrics(
                {
                    "active": True,
                    "used_for_training": True,
                    "metrics_path": "runs/frontier/transformer_answer_metrics.json",
                    "frontier_run_id": "frontier",
                    "final_comparison": {
                        "available": True,
                        "passed": True,
                    },
                }
            )
        )

        self.assertEqual(report["status"], "blocked_before_quality_metrics")
        self.assertIn(
            "direct_answer_frontier_reference_purity",
            report["failed_constraints"],
        )


def _metrics(reference: dict[str, Any]) -> dict[str, Any]:
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
        "backend": transformer_backend_metadata(
            seed=17,
            tokenizer_type="char",
        ),
        "direct_answer": {
            "direct_answer_branch_context_gate": {"passed": True},
            "direct_answer_frontier_reference": reference,
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
