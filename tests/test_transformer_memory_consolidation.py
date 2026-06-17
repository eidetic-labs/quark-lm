from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from transformer_char_model_test_support import (
    branch_diversity_snapshot_collapsed_profile_names,
    branch_diversity_snapshot_profile_diversity_delta,
    branch_diversity_snapshot_target_coverage_delta,
    parse_args,
    train_transformer_answers,
)


class TransformerMemoryConsolidationTest(unittest.TestCase):
    def test_profile_scale_memory_consolidation_frontier_mode_consumes_source_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_plan = Path(temp) / "source_memory_consolidation_plan.json"
            with source_plan.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "kind": "memory_consolidation_plan",
                        "summary": {
                            "collapsed_memory_backed_profiles": [
                                "owner",
                                "paraphrases",
                                "glossary",
                            ],
                            "top_priority_profiles": [
                                "owner",
                                "paraphrases",
                                "glossary",
                                "admission_paraphrases",
                                "admissions",
                            ],
                            "memory_backed_failed_profiles": 9,
                            "retrieval_exact_rate": 1.0,
                        },
                        "profile_priorities": [
                            {"profile": "owner", "priority_score": 5.075},
                            {"profile": "paraphrases", "priority_score": 5.0},
                            {"profile": "glossary", "priority_score": 4.333333},
                        ],
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-memory-"
                            "consolidation-frontier-screen"
                        )
                    ),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-eval-every",
                    "1",
                    "--direct-answer-mode",
                    (
                        "branch-context-profile-baseline-floor-diversity-"
                        "branch-stable-coverage-recovery-branch-diversity-"
                        "collapsed-profile-binding-remaining-profile-owner-"
                        "paraphrase-memory-consolidation-frontier-profile-"
                        "scale-calibrated-sequential-profile-stabilization-"
                        "unlikelihood"
                    ),
                    "--memory-consolidation-source-plan",
                    str(source_plan),
                    "--memory-consolidation-max-profiles",
                    "3",
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-branch-batch-size",
                    "2",
                    "--direct-answer-hard-negatives",
                    "1",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "80",
                ]
            )

            metrics = train_transformer_answers(args)

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

    def test_profile_scale_memory_consolidation_missing_first_token_mode_records_token_pressure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_plan = Path(temp) / "source_memory_consolidation_plan.json"
            with source_plan.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "kind": "memory_consolidation_plan",
                        "summary": {
                            "collapsed_memory_backed_profiles": [
                                "owner",
                                "paraphrases",
                                "glossary",
                            ],
                            "top_priority_profiles": [
                                "owner",
                                "paraphrases",
                                "glossary",
                            ],
                            "memory_backed_failed_profiles": 9,
                            "retrieval_exact_rate": 1.0,
                        },
                        "profile_priorities": [
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
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-memory-"
                            "consolidation-missing-first-token-screen"
                        )
                    ),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-eval-every",
                    "1",
                    "--direct-answer-mode",
                    (
                        "branch-context-profile-baseline-floor-diversity-"
                        "branch-stable-coverage-recovery-branch-diversity-"
                        "collapsed-profile-binding-remaining-profile-owner-"
                        "paraphrase-memory-consolidation-missing-first-token-"
                        "frontier-profile-scale-calibrated-sequential-profile-"
                        "stabilization-unlikelihood"
                    ),
                    "--memory-consolidation-source-plan",
                    str(source_plan),
                    "--memory-consolidation-max-profiles",
                    "3",
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-branch-batch-size",
                    "2",
                    "--direct-answer-hard-negatives",
                    "1",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "80",
                ]
            )

            metrics = train_transformer_answers(args)

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

    def test_profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_mode_records_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_plan = Path(temp) / "source_memory_consolidation_plan.json"
            with source_plan.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "kind": "memory_consolidation_plan",
                        "summary": {
                            "collapsed_memory_backed_profiles": [
                                "owner",
                                "paraphrases",
                                "learning",
                            ],
                            "top_priority_profiles": [
                                "owner",
                                "paraphrases",
                                "learning",
                                "admissions",
                            ],
                            "memory_backed_failed_profiles": 9,
                            "retrieval_exact_rate": 1.0,
                        },
                        "profile_priorities": [
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
                                "missing_target_tokens": [
                                    {"value": "y", "count": 3},
                                ],
                            },
                        ],
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-memory-consolidation-"
                            "remaining-collapsed-missing-first-token-screen"
                        )
                    ),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-eval-every",
                    "1",
                    "--direct-answer-mode",
                    (
                        "branch-context-profile-baseline-floor-diversity-"
                        "branch-stable-coverage-recovery-branch-diversity-"
                        "collapsed-profile-binding-remaining-profile-owner-"
                        "paraphrase-memory-consolidation-remaining-collapsed-"
                        "missing-first-token-frontier-profile-scale-calibrated-"
                        "sequential-profile-stabilization-unlikelihood"
                    ),
                    "--memory-consolidation-source-plan",
                    str(source_plan),
                    "--memory-consolidation-max-profiles",
                    "3",
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-branch-batch-size",
                    "2",
                    "--direct-answer-hard-negatives",
                    "1",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "80",
                ]
            )

            metrics = train_transformer_answers(args)

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
            source_plan = Path(temp) / "source_memory_consolidation_plan.json"
            with source_plan.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "kind": "memory_consolidation_plan",
                        "summary": {
                            "collapsed_memory_backed_profiles": [
                                "owner",
                                "paraphrases",
                                "learning",
                            ],
                            "top_priority_profiles": [
                                "owner",
                                "paraphrases",
                                "learning",
                                "admissions",
                            ],
                            "memory_backed_failed_profiles": 9,
                            "retrieval_exact_rate": 1.0,
                        },
                        "profile_priorities": [
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
                                "missing_target_tokens": [
                                    {"value": "y", "count": 3},
                                ],
                            },
                        ],
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-memory-consolidation-"
                            "remaining-collapsed-profile-specific-"
                            "missing-first-token-screen"
                        )
                    ),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-eval-every",
                    "1",
                    "--direct-answer-mode",
                    (
                        "branch-context-profile-baseline-floor-diversity-"
                        "branch-stable-coverage-recovery-branch-diversity-"
                        "collapsed-profile-binding-remaining-profile-owner-"
                        "paraphrase-memory-consolidation-remaining-collapsed-"
                        "profile-specific-missing-first-token-frontier-profile-"
                        "scale-calibrated-sequential-profile-stabilization-"
                        "unlikelihood"
                    ),
                    "--memory-consolidation-source-plan",
                    str(source_plan),
                    "--memory-consolidation-max-profiles",
                    "3",
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-branch-batch-size",
                    "2",
                    "--direct-answer-hard-negatives",
                    "1",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "80",
                ]
            )

            metrics = train_transformer_answers(args)

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

    def test_branch_diversity_target_coverage_delta_records_profile_gains(
        self,
    ) -> None:
        baseline = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 3,
                        "target_token_coverage": 1.0 / 3.0,
                    }
                },
            }
        }
        snapshot = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.5,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 3,
                        "target_token_coverage": 1.0 / 3.0,
                    }
                },
            }
        }

        delta = branch_diversity_snapshot_target_coverage_delta(snapshot, baseline)

        self.assertEqual(delta["improved_profile_count"], 1)
        self.assertEqual(delta["regressed_profile_count"], 0)
        self.assertEqual(delta["tied_profile_count"], 1)
        self.assertAlmostEqual(delta["min_delta"], 1.0 / 12.0)
        self.assertAlmostEqual(delta["average_delta"], 0.125)

    def test_branch_diversity_collapsed_profile_delta_tracks_targeted_gain(
        self,
    ) -> None:
        baseline = {
            "branch_diversity_target": {
                "blocking_evals": [
                    {"name": "owner", "collapsed": True},
                    {"name": "qa", "collapsed": False},
                ]
            },
            "branch_profiles": {
                "owner": {
                    "diversity": {
                        "predicted_unique": 1,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_rate": 1.0,
                    }
                },
                "qa": {
                    "diversity": {
                        "predicted_unique": 2,
                        "target_token_coverage": 0.25,
                        "dominant_predicted_rate": 0.5,
                    }
                },
            },
        }
        snapshot = {
            "branch_profiles": {
                "owner": {
                    "diversity": {
                        "predicted_unique": 2,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_rate": 0.75,
                    }
                },
                "qa": {
                    "diversity": {
                        "predicted_unique": 2,
                        "target_token_coverage": 0.25,
                        "dominant_predicted_rate": 0.5,
                    }
                },
            }
        }

        collapsed_profiles = branch_diversity_snapshot_collapsed_profile_names(
            baseline
        )
        delta = branch_diversity_snapshot_profile_diversity_delta(
            snapshot,
            baseline,
            collapsed_profiles,
        )

        self.assertEqual(collapsed_profiles, ["owner"])
        self.assertEqual(delta["improved_profile_count"], 1)
        self.assertEqual(delta["regressed_profile_count"], 0)
        self.assertEqual(
            delta["improved_profiles"][0]["predicted_unique_delta"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
