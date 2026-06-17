from __future__ import annotations

import tempfile
import unittest

from support.profile_scale_modes import train_profile_scale_mode_screen


class TransformerProfileScaleRemainingBindingModeTest(unittest.TestCase):
    def test_profile_scale_remaining_profile_binding_frontier_mode_records_priority_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_profile_scale_mode_screen(
                temp,
                (
                    "baseline-floor-profile-scale-remaining-profile-"
                    "binding-frontier-screen"
                ),
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "collapsed-profile-binding-remaining-profile-"
                    "frontier-profile-scale-calibrated-sequential-"
                    "profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_remaining_profile_binding_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_target_profiles"],
            ["learning", "owner", "paraphrases"],
        )
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_source_labels"],
            ["color", "learning", "owner", "place", "training_data"],
        )
        self.assertTrue(
            guard["profile_scale_remaining_profile_binding_source_profiles"]
        )
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_source_profiles"],
            replay_plan["remaining_profile_binding_source_profiles"],
        )
        self.assertEqual(
            guard[
                "profile_scale_remaining_profile_binding_prioritized_attempts"
            ],
            guard[
                "profile_scale_remaining_profile_binding_prioritized_acceptances"
            ]
            + guard[
                "profile_scale_remaining_profile_binding_prioritized_rejections"
            ],
        )
        if guard[
            "profile_scale_remaining_profile_binding_prioritized_acceptances"
        ]:
            self.assertTrue(
                guard["profile_scale_remaining_profile_binding_probe_sample"]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_remaining_profile_binding_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
