import unittest
from typing import Any

from transformer_baseline_floor_profile_outcomes import evaluate_baseline_floor_profile_outcome


class TransformerBaselineFloorProfileOutcomesTest(unittest.TestCase):
    def test_diversity_improvement_without_coverage_frontier(self) -> None:
        outcome = evaluate_baseline_floor_profile_outcome(
            profile_probe_snapshot={"score": (2.0, 1.0)},
            direct_baseline={},
            profile_base_snapshot=None,
            profile_base_score=(1.0, 1.0),
            diversity_active=True,
            coverage_frontier_active=False,
            coverage_prep_frontier_active=False,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda snapshot: snapshot["score"],
        )

        self.assertTrue(outcome.floor_preserved)
        self.assertEqual(outcome.profile_score, (2.0, 1.0))
        self.assertEqual(outcome.diversity_outcome, "improved")
        self.assertEqual(outcome.diversity_rejection_reason, "")
        self.assertEqual(outcome.coverage_outcome, "not_active")
        self.assertFalse(outcome.coverage_prep_accepted)

    def test_coverage_regression_records_rejection_reason(self) -> None:
        outcome = evaluate_baseline_floor_profile_outcome(
            profile_probe_snapshot={},
            direct_baseline={},
            profile_base_snapshot={},
            profile_base_score=None,
            diversity_active=False,
            coverage_frontier_active=True,
            coverage_prep_frontier_active=True,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 1,
                "improved_profile_count": 0,
            },
        )

        self.assertEqual(outcome.coverage_outcome, "regressed")
        self.assertEqual(outcome.coverage_rejection_reason, "coverage_regression")
        self.assertEqual(
            outcome.coverage_delta,
            {"regressed_profile_count": 1, "improved_profile_count": 0},
        )
        self.assertFalse(outcome.coverage_prep_accepted)

    def test_coverage_prep_accepts_tied_coverage_after_diversity_gain(self) -> None:
        outcome = evaluate_baseline_floor_profile_outcome(
            profile_probe_snapshot={"score": (2.0,)},
            direct_baseline={},
            profile_base_snapshot={},
            profile_base_score=(1.0,),
            diversity_active=True,
            coverage_frontier_active=True,
            coverage_prep_frontier_active=True,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda snapshot: snapshot["score"],
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
        )

        self.assertEqual(outcome.diversity_outcome, "improved")
        self.assertEqual(outcome.coverage_outcome, "tied")
        self.assertEqual(outcome.coverage_rejection_reason, "coverage_tie")
        self.assertTrue(outcome.coverage_prep_accepted)

    def test_floor_regression_blocks_diversity_and_coverage_acceptance(self) -> None:
        score_calls: list[dict[str, Any]] = []

        outcome = evaluate_baseline_floor_profile_outcome(
            profile_probe_snapshot={"score": (2.0,)},
            direct_baseline={},
            profile_base_snapshot={},
            profile_base_score=(1.0,),
            diversity_active=True,
            coverage_frontier_active=True,
            coverage_prep_frontier_active=True,
            preserves_target_coverage=lambda _snapshot, _baseline: False,
            snapshot_score=lambda snapshot: (
                score_calls.append(snapshot) or snapshot["score"]
            ),
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
        )

        self.assertFalse(outcome.floor_preserved)
        self.assertEqual(score_calls, [{"score": (2.0,)}])
        self.assertEqual(outcome.diversity_outcome, "floor_regressed")
        self.assertEqual(outcome.diversity_rejection_reason, "floor_regression")
        self.assertEqual(outcome.coverage_outcome, "floor_regressed")
        self.assertEqual(outcome.coverage_rejection_reason, "floor_regression")
        self.assertFalse(outcome.coverage_prep_accepted)


if __name__ == "__main__":
    unittest.main()
