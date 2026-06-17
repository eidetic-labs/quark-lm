from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from transformer_char_model_test_support import (
    CharTokenizer,
    branch_diversity_profile_delta_has_coverage_gain,
    memory_consolidation_missing_first_token_values,
    memory_consolidation_source_plan_targets,
    missing_first_token_anchor_batch,
    missing_first_token_ids_by_profile,
    parse_args,
    profile_specific_missing_first_token_target_map,
    profile_specific_missing_first_token_targets,
    remaining_profile_binding_profile_order,
    remaining_profile_binding_source_labels,
    train_transformer_answers,
)


class TransformerProfileScaleModesTest(unittest.TestCase):
    def test_profile_scale_coverage_recovery_frontier_mode_records_retry_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / "baseline-floor-profile-scale-coverage-recovery-frontier-screen"
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
                        "coverage-recovery-frontier-profile-scale-calibrated-"
                        "sequential-profile-stabilization-unlikelihood"
                    ),
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
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-branch-stable-"
                            "coverage-recovery-frontier-screen"
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
                        "branch-stable-coverage-recovery-frontier-"
                        "profile-scale-calibrated-sequential-profile-"
                        "stabilization-unlikelihood"
                    ),
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
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-branch-diversity-"
                            "recovery-frontier-screen"
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
                        "frontier-profile-scale-calibrated-sequential-"
                        "profile-stabilization-unlikelihood"
                    ),
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

    def test_profile_scale_collapsed_profile_binding_frontier_mode_records_binding_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-collapsed-profile-"
                            "binding-frontier-screen"
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
                        "collapsed-profile-binding-frontier-profile-scale-"
                        "calibrated-sequential-profile-stabilization-"
                        "unlikelihood"
                    ),
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

    def test_remaining_profile_binding_prioritizes_source_profile_labels(
        self,
    ) -> None:
        source_labels = remaining_profile_binding_source_labels(
            ["learning", "owner", "paraphrases"]
        )
        groups = {
            "fact:self": [
                ([0], 1, 0, "fact:self"),
                ([0], 2, 0, "fact:self"),
            ],
            "fact:owner": [
                ([0], 1, 0, "fact:owner"),
                ([0], 2, 0, "fact:owner"),
            ],
            "qa:learning": [
                ([0], 3, 0, "qa:learning"),
                ([0], 4, 0, "qa:learning"),
            ],
            "unknown:place": [
                ([0], 5, 0, "unknown:place"),
                ([0], 6, 0, "unknown:place"),
            ],
        }

        ordered = remaining_profile_binding_profile_order(
            groups,
            ["learning", "owner", "paraphrases"],
        )

        self.assertEqual(
            source_labels,
            ["color", "learning", "owner", "place", "training_data"],
        )
        self.assertEqual(
            [profile for profile, _anchors in ordered[:3]],
            ["qa:learning", "fact:owner", "unknown:place"],
        )
        self.assertEqual(ordered[-1][0], "fact:self")

    def test_remaining_profile_binding_maps_memory_consolidation_targets(
        self,
    ) -> None:
        source_labels = remaining_profile_binding_source_labels(
            ["owner", "paraphrases", "heldout", "qa", "glossary"]
        )

        self.assertEqual(
            source_labels,
            ["color", "glossary", "owner", "place", "training_data"],
        )

    def test_missing_first_token_helpers_use_source_plan_targets(
        self,
    ) -> None:
        source_plan = {
            "kind": "memory_consolidation_plan",
            "profile_priorities": [
                {
                    "profile": "owner",
                    "missing_target_tokens": [
                        {"value": "u", "count": 2},
                        {"value": "a", "count": 1},
                        {"value": "u", "count": 1},
                    ],
                },
                {
                    "profile": "qa",
                    "missing_target_tokens": [{"value": "g", "count": 1}],
                },
                {
                    "profile": "self",
                    "missing_target_tokens": [{"value": "s", "count": 1}],
                },
            ],
        }
        tokenizer = CharTokenizer(["<pad>", "a", "g", "n", "u"])

        values = memory_consolidation_missing_first_token_values(
            source_plan,
            ["owner", "qa"],
        )
        ids_by_profile = missing_first_token_ids_by_profile(tokenizer, values)
        target_ids = {
            token_id
            for token_ids in ids_by_profile.values()
            for token_id in token_ids
        }
        batch = missing_first_token_anchor_batch(
            [
                ([0], tokenizer.stoi["u"], tokenizer.stoi["n"], "fact:owner"),
                ([0], tokenizer.stoi["a"], tokenizer.stoi["n"], "fact:owner"),
                ([0], tokenizer.stoi["n"], tokenizer.stoi["n"], "fact:owner"),
                ([0], tokenizer.stoi["g"], tokenizer.stoi["n"], "qa:place"),
            ],
            target_ids,
            random.Random(7),
            8,
        )

        self.assertEqual(values, {"owner": ["u", "a"], "qa": ["g"]})
        self.assertEqual(
            ids_by_profile,
            {
                "owner": [tokenizer.stoi["u"], tokenizer.stoi["a"]],
                "qa": [tokenizer.stoi["g"]],
            },
        )
        self.assertEqual(
            sorted(target for _context, target, _predicted, _profile in batch),
            [tokenizer.stoi["a"], tokenizer.stoi["g"], tokenizer.stoi["u"]],
        )
        self.assertTrue(
            branch_diversity_profile_delta_has_coverage_gain(
                {
                    "profiles": [
                        {"profile": "owner", "coverage_delta": 0.0},
                        {"profile": "qa", "coverage_delta": 0.125},
                    ]
                }
            )
        )

    def test_memory_consolidation_source_plan_can_require_collapsed_targets(
        self,
    ) -> None:
        source_plan = {
            "kind": "memory_consolidation_plan",
            "summary": {
                "top_priority_profiles": ["owner", "paraphrases"],
            },
            "profile_priorities": [
                {"profile": "owner"},
                {"profile": "paraphrases"},
            ],
        }

        _summary, targets, top_priorities, collapsed = (
            memory_consolidation_source_plan_targets(source_plan, 2)
        )

        self.assertEqual(targets, ["owner", "paraphrases"])
        self.assertEqual(top_priorities, ["owner", "paraphrases"])
        self.assertEqual(collapsed, [])
        with self.assertRaisesRegex(ValueError, "collapsed_memory_backed_profiles"):
            memory_consolidation_source_plan_targets(
                source_plan,
                2,
                require_collapsed_targets=True,
            )

    def test_profile_specific_missing_first_token_targets_follow_source_labels(
        self,
    ) -> None:
        ids_by_profile = {
            "owner": [1, 2],
            "paraphrases": [2, 3],
            "learning": [4],
        }
        target_profiles = ["owner", "paraphrases", "learning"]

        self.assertEqual(
            profile_specific_missing_first_token_targets(
                "fact:owner",
                target_profiles,
                ids_by_profile,
            ),
            ["owner", "paraphrases"],
        )
        self.assertEqual(
            profile_specific_missing_first_token_targets(
                "fact:learning",
                target_profiles,
                ids_by_profile,
            ),
            ["learning"],
        )
        self.assertEqual(
            profile_specific_missing_first_token_targets(
                "bridge:place",
                target_profiles,
                ids_by_profile,
            ),
            ["paraphrases"],
        )
        self.assertEqual(
            profile_specific_missing_first_token_target_map(
                target_profiles,
                ids_by_profile,
            ),
            {
                "color": ["paraphrases"],
                "learning": ["learning"],
                "owner": ["owner", "paraphrases"],
                "place": ["paraphrases"],
                "training_data": ["paraphrases"],
            },
        )

    def test_profile_scale_remaining_profile_binding_frontier_mode_records_priority_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-remaining-profile-"
                            "binding-frontier-screen"
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
                        "collapsed-profile-binding-remaining-profile-"
                        "frontier-profile-scale-calibrated-sequential-"
                        "profile-stabilization-unlikelihood"
                    ),
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
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / (
                            "baseline-floor-profile-scale-owner-paraphrase-"
                            "binding-frontier-screen"
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
                        "collapsed-profile-binding-remaining-profile-"
                        "owner-paraphrase-frontier-profile-scale-calibrated-"
                        "sequential-profile-stabilization-unlikelihood"
                    ),
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
