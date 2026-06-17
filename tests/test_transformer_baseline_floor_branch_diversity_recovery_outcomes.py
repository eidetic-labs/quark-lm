import unittest

from transformer_baseline_floor_recovery_outcomes import (
    evaluate_baseline_floor_branch_diversity_recovery_outcome,
)


class TransformerBaselineFloorBranchDiversityRecoveryOutcomesTest(unittest.TestCase):
    def test_accepts_score_gain(self) -> None:
        outcome = evaluate_baseline_floor_branch_diversity_recovery_outcome(
            floor_preserved=True,
            recovery_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
        )

        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.outcome, "branch_diversity_improved")
        self.assertEqual(outcome.rejection_reason, "")

    def test_rejects_floor_regression(self) -> None:
        outcome = evaluate_baseline_floor_branch_diversity_recovery_outcome(
            floor_preserved=False,
            recovery_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
        )

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.outcome, "floor_regressed")
        self.assertEqual(outcome.rejection_reason, "floor_regression")

    def test_rejects_coverage_regression(self) -> None:
        outcome = evaluate_baseline_floor_branch_diversity_recovery_outcome(
            floor_preserved=True,
            recovery_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 1,
                "improved_profile_count": 0,
            },
        )

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.outcome, "coverage_regressed")
        self.assertEqual(outcome.rejection_reason, "coverage_regression")

    def test_rejects_score_tie_or_regression(self) -> None:
        tie = evaluate_baseline_floor_branch_diversity_recovery_outcome(
            floor_preserved=True,
            recovery_score=(1.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
        )
        regression = evaluate_baseline_floor_branch_diversity_recovery_outcome(
            floor_preserved=True,
            recovery_score=(0.5,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
        )

        self.assertFalse(tie.accepted)
        self.assertEqual(tie.outcome, "score_tied")
        self.assertEqual(tie.rejection_reason, "score_tie")
        self.assertFalse(regression.accepted)
        self.assertEqual(regression.outcome, "score_regressed")
        self.assertEqual(regression.rejection_reason, "score_regression")


if __name__ == "__main__":
    unittest.main()
