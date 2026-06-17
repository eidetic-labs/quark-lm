from __future__ import annotations

import tempfile
import unittest

from support.memory_consolidation_modes import (
    train_memory_consolidation_mode_screen,
    write_memory_consolidation_source_plan,
)


class TransformerMemoryConsolidationTest(unittest.TestCase):
    def test_profile_scale_memory_consolidation_frontier_mode_consumes_source_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_plan = write_memory_consolidation_source_plan(
                temp,
                collapsed_profiles=["owner", "paraphrases", "glossary"],
                top_priority_profiles=[
                    "owner",
                    "paraphrases",
                    "glossary",
                    "admission_paraphrases",
                    "admissions",
                ],
                profile_priorities=[
                    {"profile": "owner", "priority_score": 5.075},
                    {"profile": "paraphrases", "priority_score": 5.0},
                    {"profile": "glossary", "priority_score": 4.333333},
                ],
            )
            metrics = train_memory_consolidation_mode_screen(
                temp,
                run_name=(
                    "baseline-floor-profile-scale-memory-consolidation-"
                    "frontier-screen"
                ),
                direct_answer_mode=(
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "collapsed-profile-binding-remaining-profile-owner-"
                    "paraphrase-memory-consolidation-frontier-profile-"
                    "scale-calibrated-sequential-profile-stabilization-"
                    "unlikelihood"
                ),
                source_plan=source_plan,
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        consolidation_plan = metrics["memory_consolidation_plan"]
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_memory_consolidation_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            direct_answer["direct_answer_memory_consolidation_source_plan"],
            str(source_plan),
        )
        self.assertEqual(
            direct_answer["direct_answer_memory_consolidation_target_profiles"],
            ["owner", "paraphrases", "glossary"],
        )
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_target_profiles"],
            ["owner", "paraphrases", "glossary"],
        )
        self.assertEqual(
            guard["profile_scale_memory_consolidation_target_profiles"],
            ["owner", "paraphrases", "glossary"],
        )
        self.assertEqual(
            guard["profile_scale_memory_consolidation_source_plan_path"],
            str(source_plan),
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_collapsed_memory_backed_profiles"
            ],
            ["owner", "paraphrases", "glossary"],
        )
        self.assertEqual(
            guard["profile_scale_memory_consolidation_consumed_profile_count"],
            3,
        )
        self.assertIn(
            "glossary",
            guard["profile_scale_remaining_profile_binding_source_labels"],
        )
        self.assertEqual(
            replay_plan["memory_consolidation_target_profiles"],
            ["owner", "paraphrases", "glossary"],
        )
        self.assertEqual(
            replay_plan["memory_consolidation_consumed_profile_count"],
            3,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_prioritized_attempts"
            ],
            guard[
                "profile_scale_memory_consolidation_prioritized_acceptances"
            ]
            + guard[
                "profile_scale_memory_consolidation_prioritized_rejections"
            ],
        )
        if guard[
            "profile_scale_memory_consolidation_prioritized_acceptances"
        ]:
            self.assertTrue(
                guard["profile_scale_memory_consolidation_probe_sample"]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_memory_consolidation_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )
        self.assertGreater(
            consolidation_plan["summary"]["memory_backed_failed_profiles"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
