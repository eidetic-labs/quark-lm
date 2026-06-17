import unittest

from transformer_baseline_floor_recovery_outcomes import (
    evaluate_baseline_floor_coverage_recovery_outcome,
)


class TransformerBaselineFloorCoverageRecoveryOutcomesTest(unittest.TestCase):
    def test_rejects_branch_score_regression(self) -> None:
        outcome = evaluate_baseline_floor_coverage_recovery_outcome(
            recovery_floor_preserved=True,
            recovery_score=(1.0,),
            profile_base_score=(0.5,),
            recovery_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
            branch_stable_active=True,
            prepared_score=(2.0,),
        )

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.outcome, "branch_score_regressed")
        self.assertEqual(outcome.rejection_reason, "branch_score_regression")
        self.assertFalse(outcome.branch_stability_preserved)
        self.assertFalse(outcome.branch_stable_accepted)

    def test_accepts_coverage_gain(self) -> None:
        outcome = evaluate_baseline_floor_coverage_recovery_outcome(
            recovery_floor_preserved=True,
            recovery_score=(3.0,),
            profile_base_score=(1.0,),
            recovery_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
            branch_stable_active=True,
            prepared_score=(2.0,),
        )

        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.outcome, "gained")
        self.assertEqual(outcome.rejection_reason, "")
        self.assertTrue(outcome.branch_stability_preserved)
        self.assertTrue(outcome.branch_stable_accepted)

    def test_rejects_tie(self) -> None:
        outcome = evaluate_baseline_floor_coverage_recovery_outcome(
            recovery_floor_preserved=True,
            recovery_score=(2.0,),
            profile_base_score=(1.0,),
            recovery_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            branch_stable_active=False,
            prepared_score=None,
        )

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.outcome, "coverage_tied")
        self.assertEqual(outcome.rejection_reason, "coverage_tie")
        self.assertIsNone(outcome.branch_stability_preserved)
        self.assertFalse(outcome.branch_stable_accepted)


if __name__ == "__main__":
    unittest.main()
