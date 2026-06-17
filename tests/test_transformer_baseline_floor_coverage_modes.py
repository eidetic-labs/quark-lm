from __future__ import annotations

import tempfile
import unittest

from support.baseline_floor_modes import train_baseline_floor_mode_screen


class TransformerBaselineFloorCoverageModesTest(unittest.TestCase):
    def test_profile_scale_coverage_frontier_mode_records_coverage_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-profile-scale-coverage-frontier-screen",
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "coverage-frontier-profile-scale-calibrated-sequential-"
                    "profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_coverage_frontier_stabilization_active"])
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_coverage_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_acceptances"]
            + guard["profile_scale_coverage_frontier_rejections"],
            guard["profile_scale_coverage_frontier_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_gains"]
            + guard["profile_scale_coverage_frontier_ties"]
            + guard["profile_scale_coverage_frontier_regressions"],
            guard["profile_scale_coverage_frontier_attempts"],
        )
        if guard["profile_scale_coverage_frontier_acceptances"]:
            self.assertTrue(
                guard[
                    "profile_scale_coverage_frontier_profile_acceptance_deltas"
                ]
            )
        if guard["profile_scale_coverage_frontier_rejections"]:
            self.assertTrue(
                guard["profile_scale_coverage_frontier_rejection_reasons"]
            )
        self.assertIn("profile_scale_coverage_frontier_probe_sample", guard)
        if guard["profile_scale_coverage_frontier_attempts"]:
            self.assertTrue(guard["profile_scale_coverage_frontier_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_coverage_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_coverage_prep_frontier_mode_records_preparation_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-profile-scale-coverage-prep-frontier-screen",
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "coverage-prep-frontier-profile-scale-calibrated-"
                    "sequential-profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard["profile_scale_coverage_prep_frontier_stabilization_active"]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_acceptances"]
            + guard["profile_scale_coverage_prep_frontier_rejections"],
            guard["profile_scale_coverage_prep_frontier_attempts"],
        )
        self.assertLessEqual(
            guard["profile_scale_coverage_prep_frontier_gain_acceptances"]
            + guard["profile_scale_coverage_prep_frontier_preparations"],
            guard["profile_scale_coverage_prep_frontier_acceptances"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_attempts"],
            guard["profile_scale_coverage_prep_frontier_attempts"],
        )
        self.assertIn("profile_scale_coverage_prep_frontier_probe_sample", guard)
        if guard["profile_scale_coverage_prep_frontier_attempts"]:
            self.assertTrue(
                guard["profile_scale_coverage_prep_frontier_probe_sample"]
            )
        if guard["profile_scale_coverage_prep_frontier_acceptances"]:
            self.assertTrue(
                guard[
                    "profile_scale_coverage_prep_frontier_profile_acceptance_outcomes"
                ]
            )
        if guard["profile_scale_coverage_prep_frontier_rejections"]:
            self.assertTrue(
                guard["profile_scale_coverage_prep_frontier_rejection_reasons"]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_coverage_prep_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
