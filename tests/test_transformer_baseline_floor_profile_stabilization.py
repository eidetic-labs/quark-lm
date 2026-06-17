from __future__ import annotations

import tempfile
import unittest

from support.baseline_floor_modes import train_baseline_floor_mode_screen


class TransformerBaselineFloorProfileStabilizationTest(unittest.TestCase):
    def test_profile_targeted_stabilization_mode_records_full_floor_surface(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-profile-targeted-screen",
                (
                    "branch-context-profile-baseline-floor-profile-targeted-"
                    "stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_targeted_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_targeted_stabilization_active"])
        self.assertEqual(
            guard["stabilization_anchor_batch_size"],
            guard["stabilization_anchor_count"],
        )
        self.assertEqual(
            replay_plan["baseline_floor_stabilization_anchor_batch_size"],
            replay_plan["baseline_floor_stabilization_anchor_count"],
        )
        self.assertEqual(
            guard["stabilization_profile_target_count"],
            replay_plan["baseline_floor_stabilization_profile_target_count"],
        )
        self.assertEqual(
            guard["stabilization_anchor_profile_counts"],
            replay_plan["baseline_floor_stabilization_anchor_profile_counts"],
        )
        if guard["rejected_attempts"]:
            self.assertIn(
                "profile_targeted_stabilization",
                guard["rejected_update_shape_counts"],
            )

    def test_sequential_stabilization_mode_records_profile_repairs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-sequential-screen",
                (
                    "branch-context-profile-baseline-floor-sequential-profile-"
                    "stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_sequential_stabilization_active"]
        )
        self.assertTrue(guard["sequential_stabilization_active"])
        self.assertEqual(
            guard["stabilization_profile_group_count"],
            len(guard["stabilization_anchor_profile_counts"]),
        )
        self.assertEqual(
            replay_plan["baseline_floor_stabilization_profile_group_count"],
            guard["stabilization_profile_group_count"],
        )
        self.assertGreaterEqual(
            guard["sequential_profile_attempts"],
            guard["stabilization_profile_group_count"],
        )
        self.assertEqual(
            guard["sequential_profile_acceptances"]
            + guard["sequential_profile_rejections"],
            guard["sequential_profile_attempts"],
        )
        self.assertGreaterEqual(
            guard["sequential_profile_records"],
            guard["sequential_profile_attempts"],
        )
        self.assertIn("sequential_profile_probe_sample", guard)
        if guard["sequential_profile_attempts"]:
            self.assertTrue(guard["sequential_profile_probe_sample"])

    def test_calibrated_sequential_stabilization_mode_records_extended_scales(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-calibrated-screen",
                (
                    "branch-context-profile-baseline-floor-calibrated-"
                    "sequential-profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_calibrated_sequential_stabilization_active"
            ]
        )
        self.assertTrue(guard["sequential_stabilization_active"])
        self.assertTrue(guard["calibrated_sequential_stabilization_active"])
        self.assertIn(0.0001, guard["learning_rate_scales"])
        self.assertEqual(
            replay_plan["adaptive_learning_rate_scales"],
            guard["learning_rate_scales"],
        )
        self.assertEqual(guard["calibrated_min_learning_rate_scale"], 0.0001)
        self.assertEqual(
            replay_plan[
                "baseline_floor_calibrated_sequential_stabilization_active"
            ],
            True,
        )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn("calibrated_sequential_profile_stabilization", shape_counts)


if __name__ == "__main__":
    unittest.main()
