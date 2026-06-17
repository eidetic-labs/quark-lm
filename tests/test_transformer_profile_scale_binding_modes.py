from __future__ import annotations

import tempfile
import unittest

from support.profile_scale_modes import train_profile_scale_mode_screen


class TransformerProfileScaleBindingModesTest(unittest.TestCase):
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

    def test_profile_scale_owner_paraphrase_binding_frontier_mode_records_residual_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_profile_scale_mode_screen(
                temp,
                (
                    "baseline-floor-profile-scale-owner-paraphrase-"
                    "binding-frontier-screen"
                ),
                (
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "collapsed-profile-binding-remaining-profile-"
                    "owner-paraphrase-frontier-profile-scale-calibrated-"
                    "sequential-profile-stabilization-unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        retrieval_memory = metrics["retrieval_memory"]
        consolidation_plan = metrics["memory_consolidation_plan"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertEqual(retrieval_memory["summary"]["exact_rate"], 1.0)
        self.assertFalse(
            retrieval_memory["dataset_exclusivity"]["external_embeddings"]
        )
        self.assertFalse(
            retrieval_memory["dataset_exclusivity"]["updates_weights"]
        )
        self.assertTrue(retrieval_memory["path"].endswith("retrieval_memory_report.json"))
        self.assertTrue(
            consolidation_plan["path"].endswith("memory_consolidation_plan.json")
        )
        self.assertGreater(
            consolidation_plan["summary"]["memory_backed_failed_profiles"],
            0,
        )
        self.assertFalse(
            consolidation_plan["dataset_exclusivity"]["updates_weights"]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_target_profiles"],
            ["owner", "paraphrases"],
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_target_profiles"],
            ["owner", "paraphrases"],
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_preserved_profiles"],
            ["learning"],
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_source_labels"],
            ["color", "owner", "place", "training_data"],
        )
        self.assertTrue(
            guard["profile_scale_owner_paraphrase_binding_source_profiles"]
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_source_profiles"],
            replay_plan["owner_paraphrase_binding_source_profiles"],
        )
        self.assertEqual(
            replay_plan["owner_paraphrase_binding_target_profiles"],
            ["owner", "paraphrases"],
        )
        self.assertEqual(
            replay_plan["owner_paraphrase_binding_preserved_profiles"],
            ["learning"],
        )
        self.assertEqual(
            guard[
                "profile_scale_owner_paraphrase_binding_prioritized_attempts"
            ],
            guard[
                "profile_scale_owner_paraphrase_binding_prioritized_acceptances"
            ]
            + guard[
                "profile_scale_owner_paraphrase_binding_prioritized_rejections"
            ],
        )
        if guard[
            "profile_scale_owner_paraphrase_binding_prioritized_acceptances"
        ]:
            self.assertTrue(
                guard["profile_scale_owner_paraphrase_binding_probe_sample"]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_owner_paraphrase_binding_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
