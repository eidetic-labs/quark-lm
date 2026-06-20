from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_cli import parse_args  # noqa: E402
from transformer_experiment import (  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
