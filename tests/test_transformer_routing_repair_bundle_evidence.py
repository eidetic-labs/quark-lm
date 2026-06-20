from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.routing_repair_bundle_metrics import (  # noqa: E402
    metrics,
    rank_collapse_metrics,
)
from transformer_constraints import transformer_constraint_report  # noqa: E402
from transformer_experiment import (  # noqa: E402
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
            metrics(diversity_passed=True, baseline_coverage=0.25, final_coverage=0.5)
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
            metrics(
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
        metrics_payload = metrics(
            diversity_passed=True,
            baseline_coverage=0.25,
            final_coverage=0.5,
        )
        metrics_payload["direct_answer"].pop("routing_repair_batch_evidence")

        checks = routing_repair_bundle_checks(metrics_payload)

        by_name = {check["name"]: check for check in checks}
        self.assertFalse(by_name["profile_balanced_branch_batches"]["passed"])

    def test_decision_records_bundle_evidence(self) -> None:
        metrics_payload = metrics(
            diversity_passed=False,
            baseline_coverage=0.25,
            final_coverage=0.25,
        )
        metrics_payload["constraint_first_promotion"] = {
            "passed": False,
            "status": "blocked_before_quality_metrics",
        }

        status, _summary, evidence = transformer_experiment_decision(metrics_payload)

        by_name = {item["name"]: item for item in evidence}
        self.assertEqual(status, "rejected")
        self.assertIn("hidden_advantage_requires_coverage_response", by_name)
        self.assertFalse(
            by_name["hidden_advantage_requires_coverage_response"]["passed"]
        )

    def test_constraint_report_includes_bundle_failures(self) -> None:
        report = transformer_constraint_report(
            metrics(
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

    def test_rank_collapse_bundle_checks_use_rank_collapse_gate_names(self) -> None:
        checks = routing_repair_bundle_checks(
            rank_collapse_metrics(
                diversity_passed=False,
                baseline_coverage=0.25,
                final_coverage=0.25,
            )
        )

        by_name = {check["name"]: check for check in checks}
        self.assertIn("rank_collapse_pressure", by_name)
        self.assertIn("rank_collapse_pressure_requires_branch_response", by_name)
        self.assertTrue(by_name["rank_collapse_pressure"]["passed"])
        self.assertFalse(
            by_name["rank_collapse_pressure_requires_branch_response"]["passed"]
        )

if __name__ == "__main__":
    unittest.main()
