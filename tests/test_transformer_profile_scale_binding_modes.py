from __future__ import annotations

import tempfile
import unittest

from support.profile_scale_modes import train_profile_scale_mode_screen


class TransformerProfileScaleCollapsedBindingModeTest(unittest.TestCase):
    def test_profile_scale_collapsed_profile_binding_frontier_mode_records_binding_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_profile_scale_mode_screen(
                temp,
                (
                    "baseline-floor-profile-scale-collapsed-profile-"
                    "binding-frontier-screen"
                ),
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "collapsed-profile-binding-frontier-profile-scale-"
                    "calibrated-sequential-profile-stabilization-"
                    "unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_collapsed_profile_binding_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            replay_plan["collapsed_profile_binding_learning_rate_scales"],
            [0.25, 0.05, 0.01],
        )
        self.assertEqual(
            guard["profile_scale_collapsed_profile_binding_learning_rate_scales"],
            [0.25, 0.05, 0.01],
        )
        self.assertEqual(
            guard["profile_scale_collapsed_profile_binding_frontier_attempts"],
            guard["profile_scale_collapsed_profile_binding_frontier_acceptances"]
            + guard["profile_scale_collapsed_profile_binding_frontier_rejections"],
        )
        self.assertEqual(
            guard["profile_scale_collapsed_profile_binding_frontier_candidates"],
            guard["profile_scale_collapsed_profile_binding_frontier_acceptances"]
            + guard[
                "profile_scale_collapsed_profile_binding_frontier_fallback_acceptances"
            ],
        )
        if guard["profile_scale_collapsed_profile_binding_frontier_attempts"]:
            self.assertGreater(
                guard["profile_scale_collapsed_profile_binding_frontier_records"],
                0,
            )
            self.assertTrue(
                guard[
                    "profile_scale_collapsed_profile_binding_frontier_probe_sample"
                ]
            )
        if guard["profile_scale_collapsed_profile_binding_frontier_candidates"]:
            self.assertTrue(
                guard[
                    "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes"
                ]
            )
            self.assertTrue(
                guard[
                    "profile_scale_collapsed_profile_binding_frontier_profile_deltas"
                ]
            )
        if guard["profile_scale_collapsed_profile_binding_frontier_rejections"]:
            self.assertTrue(
                guard[
                    "profile_scale_collapsed_profile_binding_frontier_rejection_reasons"
                ]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_collapsed_profile_binding_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
