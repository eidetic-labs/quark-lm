import unittest
from typing import Any

from transformer_baseline_floor_profile_outcomes import evaluate_baseline_floor_profile_outcome
from transformer_baseline_floor_recovery_outcomes import (
    evaluate_baseline_floor_branch_diversity_recovery_outcome,
    evaluate_baseline_floor_collapsed_profile_binding_outcome,
    evaluate_baseline_floor_coverage_recovery_outcome,
    evaluate_baseline_floor_missing_first_token_outcome,
)


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

    def test_coverage_recovery_rejects_branch_score_regression(self) -> None:
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

    def test_coverage_recovery_accepts_coverage_gain(self) -> None:
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

    def test_coverage_recovery_rejects_tie(self) -> None:
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

    def test_branch_diversity_recovery_accepts_score_gain(self) -> None:
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

    def test_branch_diversity_recovery_rejects_floor_regression(self) -> None:
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

    def test_branch_diversity_recovery_rejects_coverage_regression(self) -> None:
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

    def test_branch_diversity_recovery_rejects_score_tie_or_regression(self) -> None:
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

    def test_collapsed_profile_binding_accepts_profile_gain(self) -> None:
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

    def test_collapsed_profile_binding_rejects_preservation_regression(self) -> None:
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

    def test_collapsed_profile_binding_rejects_regressions_before_gains(self) -> None:
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

    def test_missing_first_token_accepts_profile_coverage_gain(self) -> None:
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

    def test_missing_first_token_rejects_profile_and_score_regressions(self) -> None:
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

    def test_missing_first_token_rejects_tie(self) -> None:
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
