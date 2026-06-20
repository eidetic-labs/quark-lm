from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_backend_policy import transformer_backend_metadata
from transformer_experiment_decision import transformer_experiment_decision


class TransformerExperimentFrontierReferenceTest(unittest.TestCase):
    def test_experiment_decision_surfaces_frontier_reference_failure(self) -> None:
        status, _summary, evidence = transformer_experiment_decision(
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
                        "stability_preserved": True,
                        "score_preserved": False,
                    },
                }
            )
        )

        by_name = {item["name"]: item for item in evidence}
        self.assertEqual(status, "rejected")
        self.assertFalse(by_name["direct_answer_frontier_reference_final"]["passed"])
        self.assertTrue(by_name["direct_answer_frontier_reference_purity"]["passed"])
        self.assertEqual(
            by_name["direct_answer_frontier_reference_final"]["details"][
                "frontier_run_id"
            ],
            "frontier",
        )


def _metrics(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "run-001",
        "baseline": {"step": 0},
        "final": {"step": 1},
        "training_data": "answer_model corpus-derived AnswerExample lessons",
        "closed_world_verifier": {"passed": True},
        "training_recipe": {"recipe_id": "transformer-answer:test:v0.78"},
        "sweep_plan": {"kind": "transformer_sweep_plan"},
        "replay_mixture_report": {"summary": {"passed": True}},
        "constraint_first_promotion": {
            "passed": False,
            "status": "blocked_before_quality_metrics",
        },
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "backend": transformer_backend_metadata(seed=17, tokenizer_type="char"),
        "direct_answer": {
            "direct_answer_branch_context_gate": {"passed": True},
            "direct_answer_frontier_reference": reference,
            "final": {
                "branch_diversity_target": {"passed": False},
                "branch_target_coverage_by_profile": {"qa": 0.0},
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
