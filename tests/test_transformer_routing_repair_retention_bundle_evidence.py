from __future__ import annotations

import unittest
from typing import Any

from transformer_backend_policy import transformer_backend_metadata
from transformer_experiment import (
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
    TRAINING_DATA_DESCRIPTION,
)
from transformer_routing_repair_bundle import (
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_MODE,
)
from transformer_routing_repair_bundle_evidence import routing_repair_bundle_checks


class TransformerRoutingRepairRetentionBundleEvidenceTests(unittest.TestCase):
    def test_retention_bundle_rejects_missing_retention_anchor_evidence(self) -> None:
        checks = routing_repair_bundle_checks(
            _metrics(
                baseline_rank=20.0,
                final_rank=8.0,
                final_top3=0.25,
                retention_anchor_count=0,
            )
        )

        by_name = {check["name"]: check for check in checks}
        self.assertEqual(len(checks), 7)
        self.assertTrue(by_name["retention_rank_margin_pressure"]["passed"])
        self.assertFalse(by_name["retention_anchors_recorded"]["passed"])

    def test_retention_bundle_accepts_branch_score_response(self) -> None:
        checks = routing_repair_bundle_checks(
            _metrics(
                baseline_rank=20.0,
                final_rank=8.0,
                final_top3=0.25,
                retention_anchor_count=4,
            )
        )

        by_name = {check["name"]: check for check in checks}
        self.assertTrue(by_name["retention_anchors_recorded"]["passed"])
        self.assertTrue(
            by_name["retention_rank_pressure_requires_branch_response"]["passed"]
        )


def _metrics(
    *,
    baseline_rank: float,
    final_rank: float,
    final_top3: float,
    retention_anchor_count: int,
) -> dict[str, Any]:
    return {
        "run_id": "bundle-d",
        "experiment_bundle": PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
        "baseline": {"step": 0},
        "final": {"step": 1},
        "training_data": TRAINING_DATA_DESCRIPTION,
        "closed_world_verifier": {"passed": True},
        "training_recipe": {"recipe_id": "transformer-answer:test"},
        "sweep_plan": {"kind": "transformer_sweep_plan"},
        "sweep_plan_path": "runs/bundle-d/sweep_plan.json",
        "replay_mixture_report": {"summary": {"passed": True}},
        "replay_mixture_report_path": "runs/bundle-d/replay_mixture_report.json",
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "backend": transformer_backend_metadata(seed=17, tokenizer_type="char"),
        "direct_answer": {
            "direct_answer_mode": PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_MODE,
            "direct_answer_contrast_weight": 1.0,
            "direct_answer_hard_negatives": 16,
            "direct_answer_branch_context_gate": {"passed": True},
            "routing_repair_batch_evidence": _batch_evidence(
                retention_anchor_count
            ),
            "baseline": _snapshot(baseline_rank, top3_rate=0.0),
            "final": _snapshot(final_rank, top3_rate=final_top3),
        },
    }


def _snapshot(target_rank: float, *, top3_rate: float) -> dict[str, Any]:
    return {
        "branch_target_coverage_by_profile": {"qa": 0.25},
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 2,
                    "predicted_unique": 1,
                    "target_token_coverage": 0.25,
                    "dominant_predicted_rate": 1.0,
                },
                "target_rank": {
                    "avg": target_rank,
                    "top3_rate": top3_rate,
                    "top5_rate": top3_rate,
                },
            }
        },
        "branch_representation_profiles": {
            "qa": {
                "target_unique": 2,
                "target_centroid_distance": {"min": 0.2, "avg": 0.3},
                "target_centroid_margin": {
                    "min": 0.1,
                    "avg": 0.2,
                    "poorly_separated_rate": 0.0,
                },
            }
        },
        "branch_diversity_target": {
            "passed": False,
            "multi_target_profiles": 1,
            "passed_profiles": 0,
            "failed_profiles": 1,
            "min_target_token_coverage": 0.25,
            "blocking_evals": [{"profile": "qa"}],
        },
        "branch_routing_audit": {
            "representation": {
                "profile_count": 1,
                "low_separation_profile_count": 0,
                "profiles": [{"profile": "qa"}],
            }
        },
        "evals": {"qa": {"count": 1, "exact": 1}},
    }


def _batch_evidence(retention_anchor_count: int) -> dict[str, Any]:
    return {
        "bundle": PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
        "batch_builder": "profile-balanced-training-family-branch-batch",
        "step_count": 1,
        "branch_count": 3,
        "retention_anchor_count": retention_anchor_count,
        "profile_balanced_branch_batches": {
            "name": "profile_balanced_branch_batches",
            "passed": True,
            "status": "passed",
            "required_trainable_profiles": ["qa"],
            "covered_trainable_profiles": ["qa"],
            "missing_trainable_profiles": [],
            "eval_only_profiles": [],
        },
        "steps": [],
        "passed": True,
        "status": "passed",
    }


if __name__ == "__main__":
    unittest.main()
