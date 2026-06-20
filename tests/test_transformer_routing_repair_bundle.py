from __future__ import annotations

import unittest

from transformer_routing_repair_bundle import (
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE,
    PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_MODE,
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_MODE,
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_MODE,
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
    routing_repair_bundle_mode,
    routing_repair_bundle_supports_mode,
    routing_repair_bundle_failure_criteria,
    routing_repair_bundle_gates,
    routing_repair_bundle_hypothesis,
    routing_repair_bundle_notes,
)


class TransformerRoutingRepairBundleTests(unittest.TestCase):
    def test_profile_balanced_bundle_declares_required_gates(self) -> None:
        gates = routing_repair_bundle_gates(PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE)
        names = {gate["name"] for gate in gates}

        self.assertIn("profile_balanced_branch_batches", names)
        self.assertIn("hidden_projection_margin_pressure", names)
        self.assertIn("representation_separation_evidence", names)
        self.assertIn("coverage_preserving_update_guard", names)
        self.assertIn("branch_diversity_acceptance_gate", names)
        self.assertIn("hidden_advantage_requires_coverage_response", names)
        self.assertTrue(all(gate["required"] for gate in gates))

    def test_rank_bundle_declares_required_gates(self) -> None:
        gates = routing_repair_bundle_gates(
            PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE
        )
        names = {gate["name"] for gate in gates}

        self.assertIn("profile_balanced_branch_batches", names)
        self.assertIn("rank_margin_pressure", names)
        self.assertIn("representation_separation_evidence", names)
        self.assertIn("coverage_preserving_update_guard", names)
        self.assertIn("branch_diversity_acceptance_gate", names)
        self.assertIn("rank_pressure_requires_branch_response", names)
        self.assertTrue(all(gate["required"] for gate in gates))

    def test_topk_bundle_declares_required_gates(self) -> None:
        gates = routing_repair_bundle_gates(
            PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE
        )
        names = {gate["name"] for gate in gates}

        self.assertIn("profile_balanced_branch_batches", names)
        self.assertIn("topk_softmax_pressure", names)
        self.assertIn("representation_separation_evidence", names)
        self.assertIn("coverage_preserving_update_guard", names)
        self.assertIn("branch_diversity_acceptance_gate", names)
        self.assertIn("topk_pressure_requires_branch_response", names)
        self.assertTrue(all(gate["required"] for gate in gates))

    def test_retention_rank_bundle_declares_required_gates(self) -> None:
        gates = routing_repair_bundle_gates(
            PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE
        )
        names = {gate["name"] for gate in gates}

        self.assertIn("profile_balanced_branch_batches", names)
        self.assertIn("retention_rank_margin_pressure", names)
        self.assertIn("retention_anchors_recorded", names)
        self.assertIn("representation_separation_evidence", names)
        self.assertIn("coverage_preserving_update_guard", names)
        self.assertIn("branch_diversity_acceptance_gate", names)
        self.assertIn("retention_rank_pressure_requires_branch_response", names)
        self.assertTrue(all(gate["required"] for gate in gates))

    def test_rank_collapse_bundle_declares_required_gates(self) -> None:
        gates = routing_repair_bundle_gates(
            PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE
        )
        names = {gate["name"] for gate in gates}

        self.assertIn("profile_balanced_branch_batches", names)
        self.assertIn("rank_collapse_pressure", names)
        self.assertIn("representation_separation_evidence", names)
        self.assertIn("coverage_preserving_update_guard", names)
        self.assertIn("branch_diversity_acceptance_gate", names)
        self.assertIn("rank_collapse_pressure_requires_branch_response", names)
        self.assertTrue(all(gate["required"] for gate in gates))

    def test_rank_bundle_binds_profile_balanced_rank_mode(self) -> None:
        self.assertEqual(
            routing_repair_bundle_mode(PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE),
            PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE,
        )
        self.assertTrue(
            routing_repair_bundle_supports_mode(
                PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
                PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE,
            )
        )

    def test_topk_bundle_binds_profile_balanced_topk_mode(self) -> None:
        self.assertEqual(
            routing_repair_bundle_mode(PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE),
            PROFILE_BALANCED_TOPK_ROUTING_REPAIR_MODE,
        )
        self.assertTrue(
            routing_repair_bundle_supports_mode(
                PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
                PROFILE_BALANCED_TOPK_ROUTING_REPAIR_MODE,
            )
        )

    def test_retention_rank_bundle_binds_retention_rank_mode(self) -> None:
        self.assertEqual(
            routing_repair_bundle_mode(
                PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE
            ),
            PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_MODE,
        )
        self.assertTrue(
            routing_repair_bundle_supports_mode(
                PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
                PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_MODE,
            )
        )

    def test_rank_collapse_bundle_binds_rank_collapse_mode(self) -> None:
        self.assertEqual(
            routing_repair_bundle_mode(
                PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE
            ),
            PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_MODE,
        )
        self.assertTrue(
            routing_repair_bundle_supports_mode(
                PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE,
                PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_MODE,
            )
        )

    def test_unknown_bundle_adds_no_contract(self) -> None:
        self.assertEqual(routing_repair_bundle_gates(None), [])
        self.assertEqual(routing_repair_bundle_failure_criteria("other"), [])
        self.assertEqual(routing_repair_bundle_notes("other"), [])
        self.assertIsNone(routing_repair_bundle_hypothesis("other"))


if __name__ == "__main__":
    unittest.main()
