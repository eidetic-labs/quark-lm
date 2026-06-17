import unittest

from transformer_baseline_floor_recovery_outcomes import (
    evaluate_baseline_floor_missing_first_token_outcome,
)


class TransformerBaselineFloorMissingFirstTokenOutcomesTest(unittest.TestCase):
    def test_accepts_profile_coverage_gain(self) -> None:
        outcome = evaluate_baseline_floor_missing_first_token_outcome(
            floor_preserved=True,
            token_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
                "profiles": [{"coverage_delta": 0.25}],
            },
        )

        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.outcome, "missing_first_token_coverage_gained")
        self.assertEqual(outcome.rejection_reason, "")

    def test_rejects_profile_and_score_regressions(self) -> None:
        profile = evaluate_baseline_floor_missing_first_token_outcome(
            floor_preserved=True,
            token_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 1,
                "improved_profile_count": 0,
                "profiles": [{"coverage_delta": 0.25}],
            },
        )
        score = evaluate_baseline_floor_missing_first_token_outcome(
            floor_preserved=True,
            token_score=(0.5,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
                "profiles": [{"coverage_delta": 0.25}],
            },
        )

        self.assertFalse(profile.accepted)
        self.assertEqual(profile.outcome, "target_profile_regressed")
        self.assertEqual(profile.rejection_reason, "target_profile_regression")
        self.assertFalse(score.accepted)
        self.assertEqual(score.outcome, "score_regressed")
        self.assertEqual(score.rejection_reason, "score_regression")

    def test_rejects_tie(self) -> None:
        outcome = evaluate_baseline_floor_missing_first_token_outcome(
            floor_preserved=True,
            token_score=(1.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
                "profiles": [{"coverage_delta": 0.0}],
            },
        )

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.outcome, "missing_first_token_tied")
        self.assertEqual(outcome.rejection_reason, "missing_first_token_tie")


if __name__ == "__main__":
    unittest.main()
