from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_cli import parse_args  # noqa: E402
from transformer_experiment import (  # noqa: E402
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
    transformer_experiment_intent,
)


class TransformerExperimentBundleTests(unittest.TestCase):
    def test_answer_train_accepts_profile_balanced_routing_bundle(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--experiment-bundle",
                PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
            ]
        )

        self.assertEqual(args.experiment_bundle, PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE)

    def test_answer_train_accepts_profile_balanced_rank_routing_bundle(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--experiment-bundle",
                PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
            ]
        )

        self.assertEqual(
            args.experiment_bundle,
            PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
        )

    def test_answer_train_accepts_profile_balanced_topk_routing_bundle(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--experiment-bundle",
                PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
            ]
        )

        self.assertEqual(
            args.experiment_bundle,
            PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
        )

    def test_bundle_intent_adds_routing_repair_contract(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--run",
                "runs/v0.116-routing-repair",
                "--direct-answer-steps",
                "1",
                "--direct-answer-mode",
                "branch-hidden-projection-margin-unlikelihood",
                "--experiment-bundle",
                PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
            ]
        )

        intent = transformer_experiment_intent(args)
        gates = {gate["name"] for gate in intent["acceptance_gates"]}

        self.assertIn("profile_balanced_branch_batches", gates)
        self.assertIn("representation_separation_evidence", gates)
        self.assertIn("branch_diversity_acceptance_gate", gates)
        self.assertIn("Hidden-projection margin pressure", intent["hypothesis"])
        self.assertIn(
            "Hidden advantage improves while target-token coverage remains unchanged.",
            intent["failure_criteria"],
        )
        self.assertIn(
            "Experiment bundle: Bundle A, profile-balanced routing repair.",
            intent["notes"],
        )

    def test_rank_bundle_intent_adds_rank_routing_contract(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--run",
                "runs/v0.117-rank-routing-repair",
                "--direct-answer-steps",
                "1",
                "--direct-answer-mode",
                "branch-profile-balanced-rank-margin-unlikelihood",
                "--experiment-bundle",
                PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
            ]
        )

        intent = transformer_experiment_intent(args)
        gates = {gate["name"] for gate in intent["acceptance_gates"]}

        self.assertIn("profile_balanced_branch_batches", gates)
        self.assertIn("rank_margin_pressure", gates)
        self.assertIn("rank_pressure_requires_branch_response", gates)
        self.assertIn("hard-negative rank-margin pressure", intent["hypothesis"])
        self.assertIn(
            "Rank-margin pressure produces no target-rank, top-k, or coverage response.",
            intent["failure_criteria"],
        )
        self.assertIn(
            "Experiment bundle: Bundle B, profile-balanced rank routing repair.",
            intent["notes"],
        )

    def test_topk_bundle_intent_adds_topk_routing_contract(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--run",
                "runs/v0.118-topk-routing-repair",
                "--direct-answer-steps",
                "1",
                "--direct-answer-mode",
                "branch-profile-balanced-topk-softmax-unlikelihood",
                "--experiment-bundle",
                PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
            ]
        )

        intent = transformer_experiment_intent(args)
        gates = {gate["name"] for gate in intent["acceptance_gates"]}

        self.assertIn("profile_balanced_branch_batches", gates)
        self.assertIn("topk_softmax_pressure", gates)
        self.assertIn("topk_pressure_requires_branch_response", gates)
        self.assertIn("top-k softmax pressure", intent["hypothesis"])
        self.assertIn(
            "Top-k softmax pressure produces no target-rank, top-k, or coverage response.",
            intent["failure_criteria"],
        )
        self.assertIn(
            "Experiment bundle: Bundle C, profile-balanced top-k routing repair.",
            intent["notes"],
        )


if __name__ == "__main__":
    unittest.main()
