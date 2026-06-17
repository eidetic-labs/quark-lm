import unittest

from support.baseline_floor_rejection import empty_guard
from transformer_baseline_floor_rejection import (
    BaselineFloorProfileRejectionAccounting,
    record_baseline_floor_profile_attempt_rejection,
    record_baseline_floor_profile_rejection,
)


class TransformerBaselineFloorRejectionTest(unittest.TestCase):
    def test_rejection_accounting_records_counters_and_reasons(self) -> None:
        guard = empty_guard()

        reason = record_baseline_floor_profile_rejection(
            guard,
            BaselineFloorProfileRejectionAccounting(
                remaining_profile_binding_prioritized=True,
                owner_paraphrase_binding_prioritized=True,
                memory_consolidation_prioritized=True,
                diversity_active=True,
                floor_preserved=False,
                diversity_rejection_reason="floor_regression",
                frontier_active=True,
                coverage_frontier_active=True,
                coverage_outcome="regressed",
                coverage_rejection_reason="coverage_regression",
                coverage_prep_active=True,
            ),
        )

        self.assertEqual(reason, "floor_regression")
        self.assertEqual(guard["sequential_profile_rejections"], 1)
        self.assertEqual(guard["profile_scale_memory_rejections"], 1)
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_prioritized_rejections"],
            1,
        )
        self.assertEqual(guard["profile_scale_diversity_floor_rejections"], 1)
        self.assertEqual(guard["profile_scale_frontier_rejections"], 1)
        self.assertEqual(guard["profile_scale_coverage_frontier_regressions"], 1)
        self.assertEqual(
            guard["profile_scale_diversity_rejection_reasons"],
            {"floor_regression": 1},
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_rejection_reasons"],
            {"coverage_regression": 1},
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_rejection_reasons"],
            {"coverage_regression": 1},
        )

    def test_rejection_accounting_relabels_diversity_for_coverage_failure(self) -> None:
        guard = empty_guard()

        reason = record_baseline_floor_profile_rejection(
            guard,
            BaselineFloorProfileRejectionAccounting(
                diversity_active=True,
                floor_preserved=True,
                diversity_accepted=True,
                coverage_frontier_active=True,
                coverage_outcome="tied",
                coverage_prep_active=True,
            ),
        )

        self.assertEqual(reason, "coverage_frontier_rejection")
        self.assertEqual(
            guard["profile_scale_diversity_rejection_reasons"],
            {"coverage_frontier_rejection": 1},
        )
        self.assertEqual(guard["profile_scale_coverage_frontier_ties"], 1)
        self.assertEqual(
            guard["profile_scale_coverage_frontier_rejection_reasons"],
            {"not_accepted": 1},
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_rejection_reasons"],
            {"coverage_frontier_rejection": 1},
        )

    def test_profile_attempt_rejection_records_counts_and_sample(self) -> None:
        guard = empty_guard()

        reason = record_baseline_floor_profile_attempt_rejection(
            guard,
            profile="qa:owner",
            records=3,
            frontier_records=1,
            learning_rate_scale=0.5,
            scale_key="0.5",
            direct_baseline={},
            profile_probe_snapshot={},
            remaining_profile_binding_prioritized=True,
            owner_paraphrase_binding_prioritized=False,
            memory_consolidation_prioritized=False,
            diversity_active=True,
            floor_preserved=True,
            diversity_accepted=True,
            diversity_outcome="improved",
            diversity_rejection_reason="",
            profile_score=(2.0,),
            profile_base_score=(1.0,),
            frontier_active=True,
            coverage_frontier_active=True,
            coverage_delta={
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            coverage_outcome="tied",
            coverage_prep_active=True,
            coverage_prep_accepted=False,
            coverage_rejection_reason="coverage_tie_without_score_gain",
            target_coverage_diagnostics=lambda _snapshot, _baseline: {
                "worst_violation": 0.25,
                "violating_profile_count": 1,
            },
        )

        self.assertEqual(reason, "coverage_frontier_rejection")
        self.assertEqual(
            guard["profile_scale_diversity_rejection_reasons"],
            {"coverage_frontier_rejection": 1},
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_rejection_reasons"],
            {"coverage_tie_without_score_gain": 1},
        )
        self.assertEqual(
            guard["sequential_profile_rejection_counts"],
            {"qa:owner": 1},
        )
        self.assertEqual(
            guard["profile_scale_rejection_scale_counts"],
            {"0.5": 1},
        )
        sample = guard["profile_scale_probe_sample"][0]
        self.assertEqual(sample["profile"], "qa:owner")
        self.assertEqual(
            sample["diversity_rejection_reason"],
            "coverage_frontier_rejection",
        )
        self.assertEqual(
            sample["coverage_rejection_reason"],
            "coverage_tie_without_score_gain",
        )
        self.assertEqual(sample["worst_violation"], 0.25)


if __name__ == "__main__":
    unittest.main()
