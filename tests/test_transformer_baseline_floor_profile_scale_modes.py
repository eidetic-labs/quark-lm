from __future__ import annotations

import tempfile
import unittest

from support.baseline_floor_modes import train_baseline_floor_mode_screen


class TransformerBaselineFloorProfileScaleModesTest(unittest.TestCase):
    def test_profile_scale_calibrated_stabilization_mode_records_scale_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-profile-scale-screen",
                (
                    "branch-context-profile-baseline-floor-profile-scale-"
                    "calibrated-sequential-profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_calibrated_stabilization_active"])
        self.assertEqual(guard["outer_learning_rate_scales"], [1.0])
        self.assertEqual(replay_plan["outer_learning_rate_scales"], [1.0])
        self.assertIn(0.0001, guard["learning_rate_scales"])
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_calibrated_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_memory_attempts"],
            guard["sequential_profile_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_memory_acceptances"]
            + guard["profile_scale_memory_rejections"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertIn("profile_scale_probe_sample", guard)
        if guard["profile_scale_memory_attempts"]:
            self.assertTrue(guard["profile_scale_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_diversity_stabilization_mode_records_diversity_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-profile-scale-diversity-screen",
                (
                    "branch-context-profile-baseline-floor-diversity-profile-"
                    "scale-calibrated-sequential-profile-stabilization-"
                    "unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_diversity_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_calibrated_stabilization_active"])
        self.assertTrue(guard["profile_scale_diversity_stabilization_active"])
        self.assertEqual(guard["outer_learning_rate_scales"], [1.0])
        self.assertEqual(replay_plan["outer_learning_rate_scales"], [1.0])
        self.assertEqual(
            replay_plan["baseline_floor_profile_scale_diversity_stabilization_active"],
            True,
        )
        self.assertEqual(
            guard["profile_scale_diversity_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_diversity_acceptances"]
            + guard["profile_scale_diversity_rejections"],
            guard["profile_scale_diversity_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_diversity_score_improvements"]
            + guard["profile_scale_diversity_score_ties"]
            + guard["profile_scale_diversity_score_regressions"]
            + guard["profile_scale_diversity_floor_rejections"],
            guard["profile_scale_diversity_attempts"],
        )
        self.assertIn("profile_scale_diversity_probe_sample", guard)
        if guard["profile_scale_diversity_attempts"]:
            self.assertTrue(guard["profile_scale_diversity_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_frontier_stabilization_mode_records_frontier_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-profile-scale-frontier-screen",
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "frontier-profile-scale-calibrated-sequential-profile-"
                    "stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_diversity_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_frontier_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_calibrated_stabilization_active"])
        self.assertTrue(guard["profile_scale_diversity_stabilization_active"])
        self.assertTrue(guard["profile_scale_frontier_stabilization_active"])
        self.assertEqual(
            replay_plan["baseline_floor_profile_scale_frontier_stabilization_active"],
            True,
        )
        self.assertEqual(
            guard["frontier_anchor_count"],
            replay_plan["baseline_floor_frontier_anchor_count"],
        )
        self.assertGreaterEqual(guard["frontier_anchor_count"], 0)
        self.assertEqual(
            guard["profile_scale_frontier_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_frontier_acceptances"]
            + guard["profile_scale_frontier_rejections"],
            guard["profile_scale_frontier_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_diversity_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertIn("profile_scale_frontier_probe_sample", guard)
        if guard["profile_scale_frontier_attempts"]:
            self.assertTrue(guard["profile_scale_frontier_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
