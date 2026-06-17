from __future__ import annotations

import tempfile
import unittest

from support.profile_scale_modes import train_profile_scale_mode_screen


class TransformerProfileScaleRecoveryModesTest(unittest.TestCase):
    def test_profile_scale_coverage_recovery_frontier_mode_records_retry_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_profile_scale_mode_screen(
                temp,
                "baseline-floor-profile-scale-coverage-recovery-frontier-screen",
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "coverage-recovery-frontier-profile-scale-calibrated-"
                    "sequential-profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard["profile_scale_coverage_recovery_frontier_stabilization_active"]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_coverage_recovery_learning_rate_scales"],
            [1.0, 0.25, 0.05],
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_attempts"],
            guard["profile_scale_coverage_recovery_frontier_acceptances"]
            + guard["profile_scale_coverage_recovery_frontier_rejections"],
        )
        self.assertLessEqual(
            guard["profile_scale_coverage_recovery_frontier_fallback_preparations"],
            guard["profile_scale_coverage_recovery_frontier_prepared_candidates"],
        )
        self.assertIn("profile_scale_coverage_recovery_frontier_probe_sample", guard)
        if guard["profile_scale_coverage_recovery_frontier_attempts"]:
            self.assertGreater(
                guard["profile_scale_coverage_recovery_frontier_records"],
                0,
            )
            self.assertTrue(
                guard["profile_scale_coverage_recovery_frontier_probe_sample"]
            )
        if guard["profile_scale_coverage_recovery_frontier_acceptances"]:
            self.assertTrue(
                guard[
                    "profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes"
                ]
            )
        if guard["profile_scale_coverage_recovery_frontier_rejections"]:
            self.assertTrue(
                guard["profile_scale_coverage_recovery_frontier_rejection_reasons"]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_branch_stable_coverage_recovery_frontier_mode_records_stability_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_profile_scale_mode_screen(
                temp,
                (
                    "baseline-floor-profile-scale-branch-stable-"
                    "coverage-recovery-frontier-screen"
                ),
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-frontier-"
                    "profile-scale-calibrated-sequential-profile-"
                    "stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_branch_stable_coverage_recovery_frontier_checks"],
            guard["profile_scale_branch_stable_coverage_recovery_frontier_acceptances"]
            + guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_rejections"
            ],
        )
        self.assertLessEqual(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_acceptances"
            ],
            guard["profile_scale_coverage_recovery_frontier_acceptances"],
        )
        self.assertLessEqual(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_fallback_preparations"
            ],
            guard["profile_scale_coverage_recovery_frontier_prepared_candidates"],
        )
        self.assertIn(
            "profile_scale_branch_stable_coverage_recovery_frontier_probe_sample",
            guard,
        )
        if guard["profile_scale_branch_stable_coverage_recovery_frontier_checks"]:
            self.assertTrue(
                guard[
                    "profile_scale_branch_stable_coverage_recovery_frontier_probe_sample"
                ]
            )
        if guard[
            "profile_scale_branch_stable_coverage_recovery_frontier_acceptances"
        ]:
            self.assertTrue(
                guard[
                    "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes"
                ]
            )
        if guard[
            "profile_scale_branch_stable_coverage_recovery_frontier_rejections"
        ]:
            self.assertTrue(
                guard[
                    "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons"
                ]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_branch_stable_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_branch_diversity_recovery_frontier_mode_records_recovery_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_profile_scale_mode_screen(
                temp,
                (
                    "baseline-floor-profile-scale-branch-diversity-"
                    "recovery-frontier-screen"
                ),
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "frontier-profile-scale-calibrated-sequential-"
                    "profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_branch_diversity_recovery_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_branch_diversity_recovery_learning_rate_scales"],
            [0.25, 0.05, 0.01],
        )
        self.assertEqual(
            guard["profile_scale_branch_diversity_recovery_frontier_attempts"],
            guard["profile_scale_branch_diversity_recovery_frontier_acceptances"]
            + guard["profile_scale_branch_diversity_recovery_frontier_rejections"],
        )
        self.assertEqual(
            guard["profile_scale_branch_diversity_recovery_frontier_candidates"],
            guard["profile_scale_branch_diversity_recovery_frontier_acceptances"]
            + guard[
                "profile_scale_branch_diversity_recovery_frontier_fallback_acceptances"
            ],
        )
        if guard["profile_scale_branch_diversity_recovery_frontier_attempts"]:
            self.assertGreater(
                guard["profile_scale_branch_diversity_recovery_frontier_records"],
                0,
            )
            self.assertTrue(
                guard[
                    "profile_scale_branch_diversity_recovery_frontier_probe_sample"
                ]
            )
        if guard["profile_scale_branch_diversity_recovery_frontier_candidates"]:
            self.assertTrue(
                guard[
                    "profile_scale_branch_diversity_recovery_frontier_profile_acceptance_outcomes"
                ]
            )
            self.assertTrue(
                guard[
                    "profile_scale_branch_diversity_recovery_frontier_profile_score_deltas"
                ]
            )
        if guard["profile_scale_branch_diversity_recovery_frontier_rejections"]:
            self.assertTrue(
                guard[
                    "profile_scale_branch_diversity_recovery_frontier_rejection_reasons"
                ]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_branch_diversity_recovery_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
