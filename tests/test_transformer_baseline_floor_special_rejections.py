import unittest

from support.baseline_floor_rejection import empty_guard
from transformer_baseline_floor_special_rejections import (
    record_baseline_floor_branch_diversity_recovery_rejection,
    record_baseline_floor_collapsed_profile_binding_rejection,
    record_baseline_floor_coverage_recovery_rejection,
    record_baseline_floor_missing_first_token_rejection,
)


class TransformerBaselineFloorSpecialRejectionsTest(unittest.TestCase):
    def test_coverage_recovery_rejection_records_primary_and_branch_stable(self) -> None:
        guard = empty_guard()

        record_baseline_floor_coverage_recovery_rejection(
            guard,
            "branch_score_regression",
            branch_stable_active=True,
        )

        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_rejections"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_rejection_reasons"],
            {"branch_score_regression": 1},
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_rejections"
            ],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons"
            ],
            {"branch_score_regression": 1},
        )

    def test_branch_diversity_recovery_rejection_records_reason(self) -> None:
        guard = empty_guard()

        record_baseline_floor_branch_diversity_recovery_rejection(
            guard,
            "score_tie",
        )

        self.assertEqual(
            guard["profile_scale_branch_diversity_recovery_frontier_rejections"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_diversity_recovery_frontier_rejection_reasons"
            ],
            {"score_tie": 1},
        )

    def test_collapsed_profile_binding_rejection_records_reason(self) -> None:
        guard = empty_guard()

        record_baseline_floor_collapsed_profile_binding_rejection(
            guard,
            "profile_diversity_regression",
        )

        self.assertEqual(
            guard["profile_scale_collapsed_profile_binding_frontier_rejections"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_collapsed_profile_binding_frontier_rejection_reasons"
            ],
            {"profile_diversity_regression": 1},
        )

    def test_missing_first_token_rejection_records_reason(self) -> None:
        guard = empty_guard()

        record_baseline_floor_missing_first_token_rejection(
            guard,
            "target_profile_regression",
        )

        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_rejections"
            ],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_rejection_reasons"
            ],
            {"target_profile_regression": 1},
        )


if __name__ == "__main__":
    unittest.main()
