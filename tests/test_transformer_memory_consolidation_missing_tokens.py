from __future__ import annotations

import tempfile
import unittest

from support.memory_consolidation_modes import (
    train_memory_consolidation_mode_screen,
    write_memory_consolidation_source_plan,
)


class TransformerMemoryConsolidationMissingTokensTest(unittest.TestCase):
    def test_profile_scale_memory_consolidation_missing_first_token_mode_records_token_pressure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_plan = write_memory_consolidation_source_plan(
                temp,
                collapsed_profiles=["owner", "paraphrases", "glossary"],
                top_priority_profiles=["owner", "paraphrases", "glossary"],
                profile_priorities=[
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
                        "profile": "glossary",
                        "priority_score": 4.333333,
                        "missing_target_tokens": [
                            {"value": "a", "count": 24},
                            {"value": "t", "count": 4},
                        ],
                    },
                ],
            )
            metrics = train_memory_consolidation_mode_screen(
                temp,
                run_name=(
                    "baseline-floor-profile-scale-memory-consolidation-"
                    "missing-first-token-screen"
                ),
                direct_answer_mode=(
                    "branch-context-profile-baseline-floor-diversity-"
                    "branch-stable-coverage-recovery-branch-diversity-"
                    "collapsed-profile-binding-remaining-profile-owner-"
                    "paraphrase-memory-consolidation-missing-first-token-"
                    "frontier-profile-scale-calibrated-sequential-profile-"
                    "stabilization-unlikelihood"
                ),
                source_plan=source_plan,
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        expected_tokens = {
            "owner": ["u", "a"],
            "paraphrases": ["u", "g"],
            "glossary": ["a", "t"],
        }

        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_missing_first_token_consolidation_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            replay_plan[
                "baseline_floor_profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active"
            ]
        )
        self.assertEqual(
            direct_answer[
                "direct_answer_memory_consolidation_missing_first_token_target_tokens"
            ],
            expected_tokens,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_target_tokens"
            ],
            expected_tokens,
        )
        self.assertEqual(
            replay_plan[
                "memory_consolidation_missing_first_token_target_tokens"
            ],
            expected_tokens,
        )
        self.assertTrue(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_unique_target_ids"
            ]
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_attempts"
            ],
            guard[
                "profile_scale_memory_consolidation_missing_first_token_acceptances"
            ]
            + guard[
                "profile_scale_memory_consolidation_missing_first_token_rejections"
            ],
        )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_memory_consolidation_missing_first_token_frontier_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
