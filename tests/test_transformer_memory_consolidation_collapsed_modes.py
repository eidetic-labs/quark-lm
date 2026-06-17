from __future__ import annotations

import tempfile
import unittest

from support.memory_consolidation_modes import (
    train_memory_consolidation_mode_screen,
    write_memory_consolidation_source_plan,
)


class TransformerMemoryConsolidationCollapsedModesTest(unittest.TestCase):
    def test_profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_mode_records_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_plan = write_memory_consolidation_source_plan(
                temp,
                collapsed_profiles=["owner", "paraphrases", "learning"],
                top_priority_profiles=[
                    "owner",
                    "paraphrases",
                    "learning",
                    "admissions",
                ],
                profile_priorities=_collapsed_profile_priorities(),
            )
            metrics = train_memory_consolidation_mode_screen(
                temp,
                run_name=(
                    "baseline-floor-profile-scale-memory-consolidation-"
                    "remaining-collapsed-missing-first-token-screen"
                ),
                direct_answer_mode=(
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "collapsed-profile-binding-remaining-profile-owner-"
                    "paraphrase-memory-consolidation-remaining-collapsed-"
                    "missing-first-token-frontier-profile-scale-calibrated-"
                    "sequential-profile-stabilization-unlikelihood"
                ),
                source_plan=source_plan,
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        expected_targets = ["owner", "paraphrases", "learning"]
        expected_tokens = {
            "owner": ["u", "a"],
            "paraphrases": ["u", "g"],
            "learning": ["i", "w"],
        }

        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_remaining_collapsed_missing_first_token_consolidation_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            replay_plan[
                "baseline_floor_profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            direct_answer["direct_answer_memory_consolidation_target_profiles"],
            expected_targets,
        )
        self.assertEqual(
            direct_answer[
                "direct_answer_memory_consolidation_remaining_collapsed_target_profiles"
            ],
            expected_targets,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_remaining_collapsed_target_profiles"
            ],
            expected_targets,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_remaining_collapsed_source_profiles"
            ],
            expected_targets,
        )
        self.assertEqual(
            replay_plan["memory_consolidation_remaining_collapsed_target_profiles"],
            expected_targets,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_target_tokens"
            ],
            expected_tokens,
        )
        self.assertNotIn(
            "admissions",
            guard[
                "profile_scale_memory_consolidation_missing_first_token_target_tokens"
            ],
        )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_memory_consolidation_profile_specific_missing_first_token_mode_records_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_plan = write_memory_consolidation_source_plan(
                temp,
                collapsed_profiles=["owner", "paraphrases", "learning"],
                top_priority_profiles=[
                    "owner",
                    "paraphrases",
                    "learning",
                    "admissions",
                ],
                profile_priorities=_collapsed_profile_priorities(),
            )
            metrics = train_memory_consolidation_mode_screen(
                temp,
                run_name=(
                    "baseline-floor-profile-scale-memory-consolidation-"
                    "remaining-collapsed-profile-specific-"
                    "missing-first-token-screen"
                ),
                direct_answer_mode=(
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "collapsed-profile-binding-remaining-profile-owner-"
                    "paraphrase-memory-consolidation-remaining-collapsed-"
                    "profile-specific-missing-first-token-frontier-profile-"
                    "scale-calibrated-sequential-profile-stabilization-"
                    "unlikelihood"
                ),
                source_plan=source_plan,
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        expected_targets = ["owner", "paraphrases", "learning"]
        expected_map = {
            "color": ["paraphrases"],
            "learning": ["learning"],
            "owner": ["owner", "paraphrases"],
            "place": ["paraphrases"],
            "training_data": ["paraphrases"],
        }

        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_remaining_collapsed_profile_specific_missing_first_token_consolidation_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            replay_plan[
                "baseline_floor_profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            direct_answer["direct_answer_memory_consolidation_target_profiles"],
            expected_targets,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_profile_specific_missing_first_token_target_map"
            ],
            expected_map,
        )
        self.assertEqual(
            replay_plan[
                "memory_consolidation_profile_specific_missing_first_token_target_map"
            ],
            expected_map,
        )
        self.assertEqual(
            direct_answer[
                "direct_answer_memory_consolidation_profile_specific_missing_first_token_target_map"
            ],
            expected_map,
        )
        self.assertNotIn(
            "admissions",
            guard[
                "profile_scale_memory_consolidation_missing_first_token_target_tokens"
            ],
        )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


def _collapsed_profile_priorities() -> list[dict[str, object]]:
    return [
        {
            "profile": "owner",
            "priority_score": 5.075,
            "missing_target_tokens": [
                {"value": "u", "count": 2},
                {"value": "a", "count": 1},
            ],
        },
        {
            "profile": "paraphrases",
            "priority_score": 5.0,
            "missing_target_tokens": [
                {"value": "u", "count": 3},
                {"value": "g", "count": 1},
            ],
        },
        {
            "profile": "learning",
            "priority_score": 4.75,
            "missing_target_tokens": [
                {"value": "i", "count": 1},
                {"value": "w", "count": 1},
            ],
        },
        {
            "profile": "admissions",
            "priority_score": 4.0,
            "missing_target_tokens": [{"value": "y", "count": 3}],
        },
    ]


if __name__ == "__main__":
    unittest.main()
