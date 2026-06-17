import unittest

from transformer_baseline_floor_recovery_outcomes import (
    evaluate_baseline_floor_collapsed_profile_binding_outcome,
)


class TransformerBaselineFloorCollapsedProfileBindingOutcomesTest(unittest.TestCase):
    def test_accepts_profile_gain(self) -> None:
        outcome = evaluate_baseline_floor_collapsed_profile_binding_outcome(
            floor_preserved=True,
            binding_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
            owner_paraphrase_preservation_regressed=False,
        )

        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.outcome, "collapsed_profile_improved")
        self.assertEqual(outcome.rejection_reason, "")
        self.assertFalse(outcome.owner_paraphrase_preservation_failed)

    def test_rejects_preservation_regression(self) -> None:
        outcome = evaluate_baseline_floor_collapsed_profile_binding_outcome(
            floor_preserved=True,
            binding_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
            owner_paraphrase_preservation_regressed=True,
        )

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.outcome, "preserved_profile_regressed")
        self.assertEqual(
            outcome.rejection_reason,
            "owner_paraphrase_preservation_regression",
        )
        self.assertTrue(outcome.owner_paraphrase_preservation_failed)

    def test_rejects_regressions_before_gains(self) -> None:
        coverage = evaluate_baseline_floor_collapsed_profile_binding_outcome(
            floor_preserved=True,
            binding_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 1,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
            owner_paraphrase_preservation_regressed=False,
        )
        profile = evaluate_baseline_floor_collapsed_profile_binding_outcome(
            floor_preserved=True,
            binding_score=(2.0,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 1,
                "improved_profile_count": 1,
            },
            owner_paraphrase_preservation_regressed=False,
        )
        score = evaluate_baseline_floor_collapsed_profile_binding_outcome(
            floor_preserved=True,
            binding_score=(0.5,),
            base_score=(1.0,),
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
            owner_paraphrase_preservation_regressed=False,
        )

        self.assertEqual(coverage.outcome, "coverage_regressed")
        self.assertEqual(coverage.rejection_reason, "coverage_regression")
        self.assertEqual(profile.outcome, "profile_diversity_regressed")
        self.assertEqual(profile.rejection_reason, "profile_diversity_regression")
        self.assertEqual(score.outcome, "score_regressed")
        self.assertEqual(score.rejection_reason, "score_regression")


if __name__ == "__main__":
    unittest.main()
