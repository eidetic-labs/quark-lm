from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_backend_policy import transformer_backend_metadata  # noqa: E402
from transformer_constraints import transformer_constraint_report  # noqa: E402
from transformer_experiment import (  # noqa: E402
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
    TRAINING_DATA_DESCRIPTION,
    transformer_experiment_decision,
)
from transformer_routing_repair_bundle_evidence import (  # noqa: E402
    routing_repair_bundle_checks,
)


class TransformerRoutingRepairBundleEvidenceTests(unittest.TestCase):
    def test_no_checks_without_declared_bundle(self) -> None:
        self.assertEqual(routing_repair_bundle_checks({}), [])

    def test_bundle_checks_pass_with_recorded_response(self) -> None:
        checks = routing_repair_bundle_checks(
            _metrics(diversity_passed=True, baseline_coverage=0.25, final_coverage=0.5)
        )

        by_name = {check["name"]: check for check in checks}
        self.assertEqual(len(checks), 6)
        self.assertTrue(by_name["profile_balanced_branch_batches"]["passed"])
        self.assertTrue(by_name["hidden_projection_margin_pressure"]["passed"])
        self.assertTrue(by_name["representation_separation_evidence"]["passed"])
        self.assertTrue(by_name["coverage_preserving_update_guard"]["passed"])
        self.assertTrue(by_name["branch_diversity_acceptance_gate"]["passed"])
        self.assertTrue(
            by_name["hidden_advantage_requires_coverage_response"]["passed"]
        )

    def test_bundle_checks_reject_hidden_pressure_without_coverage_response(
        self,
    ) -> None:
        checks = routing_repair_bundle_checks(
            _metrics(
                diversity_passed=False,
                baseline_coverage=0.25,
                final_coverage=0.25,
            )
        )

        by_name = {check["name"]: check for check in checks}
        self.assertFalse(by_name["branch_diversity_acceptance_gate"]["passed"])
        self.assertFalse(
            by_name["hidden_advantage_requires_coverage_response"]["passed"]
        )
        self.assertTrue(by_name["coverage_preserving_update_guard"]["passed"])

    def test_bundle_checks_reject_missing_batch_evidence(self) -> None:
        metrics = _metrics(
            diversity_passed=True,
            baseline_coverage=0.25,
            final_coverage=0.5,
        )
        metrics["direct_answer"].pop("routing_repair_batch_evidence")

        checks = routing_repair_bundle_checks(metrics)

        by_name = {check["name"]: check for check in checks}
        self.assertFalse(by_name["profile_balanced_branch_batches"]["passed"])

    def test_decision_records_bundle_evidence(self) -> None:
        metrics = _metrics(
            diversity_passed=False,
            baseline_coverage=0.25,
            final_coverage=0.25,
        )
        metrics["constraint_first_promotion"] = {
            "passed": False,
            "status": "blocked_before_quality_metrics",
        }

        status, _summary, evidence = transformer_experiment_decision(metrics)

        by_name = {item["name"]: item for item in evidence}
        self.assertEqual(status, "rejected")
        self.assertIn("hidden_advantage_requires_coverage_response", by_name)
        self.assertFalse(
            by_name["hidden_advantage_requires_coverage_response"]["passed"]
        )

    def test_constraint_report_includes_bundle_failures(self) -> None:
        report = transformer_constraint_report(
            _metrics(
                diversity_passed=False,
                baseline_coverage=0.25,
                final_coverage=0.25,
            )
        )

        self.assertEqual(report["status"], "blocked_before_quality_metrics")
        self.assertIn(
            "hidden_advantage_requires_coverage_response",
            report["failed_constraints"],
        )
        self.assertIn(
            "branch_diversity_acceptance_gate",
            report["failed_constraints"],
        )


def _metrics(
    *,
    diversity_passed: bool,
    baseline_coverage: float,
    final_coverage: float,
) -> dict[str, Any]:
    return {
        "run_id": "bundle-a",
        "experiment_bundle": PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
        "baseline": {"step": 0},
        "final": {"step": 1},
        "training_data": TRAINING_DATA_DESCRIPTION,
        "closed_world_verifier": {"passed": True},
        "training_recipe": {"recipe_id": "transformer-answer:test"},
        "sweep_plan": {"kind": "transformer_sweep_plan"},
        "sweep_plan_path": "runs/bundle-a/sweep_plan.json",
        "replay_mixture_report": {"summary": {"passed": True}},
        "replay_mixture_report_path": "runs/bundle-a/replay_mixture_report.json",
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "backend": transformer_backend_metadata(seed=17, tokenizer_type="char"),
        "direct_answer": {
            "direct_answer_branch_context_gate": {"passed": True},
            "routing_repair_batch_evidence": _routing_repair_batch_evidence(),
            "baseline": _snapshot(baseline_coverage, diversity_passed=True),
            "final": _snapshot(final_coverage, diversity_passed=diversity_passed),
        },
    }


def _snapshot(coverage: float, *, diversity_passed: bool) -> dict[str, Any]:
    return {
        "branch_target_coverage_by_profile": {"qa": coverage},
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 2,
                    "target_token_coverage": coverage,
                }
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
            "passed": diversity_passed,
            "multi_target_profiles": 1,
            "passed_profiles": 1 if diversity_passed else 0,
            "failed_profiles": 0 if diversity_passed else 1,
            "blocking_evals": [] if diversity_passed else [{"profile": "qa"}],
        },
        "branch_routing_audit": {
            "representation": {
                "profile_count": 1,
                "low_separation_profile_count": 0,
                "profiles": [{"profile": "qa"}],
            },
            "logit_prior": {
                "profile_count": 1,
                "hidden_projection_profile_count": 1,
                "mixed_profile_count": 0,
                "pressure_counts": {"hidden_projection": 1},
                "profiles": [{"profile": "qa"}],
            },
        },
        "evals": {"qa": {"count": 1, "exact": 1}},
    }


def _routing_repair_batch_evidence() -> dict[str, Any]:
    return {
        "bundle": PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
        "batch_builder": "profile-balanced-training-family-branch-batch",
        "step_count": 1,
        "branch_count": 3,
        "profiles": ["glossary", "learning", "qa"],
        "required_eval_profiles": {
            "failed_profiles": ["glossary", "learning", "qa"],
            "zero_coverage_profiles": ["learning"],
            "buried_target_profiles": ["glossary", "qa"],
        },
        "profile_balanced_branch_batches": {
            "name": "profile_balanced_branch_batches",
            "passed": True,
            "status": "passed",
            "required_trainable_profiles": ["glossary", "learning", "qa"],
            "covered_trainable_profiles": ["glossary", "learning", "qa"],
            "missing_trainable_profiles": [],
            "eval_only_profiles": [],
        },
        "steps": [],
        "passed": True,
        "status": "passed",
    }


if __name__ == "__main__":
    unittest.main()
