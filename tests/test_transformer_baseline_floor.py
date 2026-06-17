from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from support.commands import parse_args, train_transformer_answers
from transformer_baseline_floor_anchor_selection import (
    baseline_floor_profile_attempt,
    baseline_floor_profile_setup,
)


class TransformerBaselineFloorTest(unittest.TestCase):
    def test_baseline_floor_profile_setup_prepares_frontier_profile_state(
        self,
    ) -> None:
        repair_anchors = [
            ([0], 1, 1, "qa:learning"),
            ([1], 2, 2, "qa:learning"),
            ([2], 3, 3, "fact:self"),
            ([3], 4, 4, "fact:owner"),
        ]
        frontier_anchors = [
            ([4], 5, 5, "qa:learning"),
            ([5], 4, 4, "fact:owner"),
        ]

        setup = baseline_floor_profile_setup(
            repair_anchors,
            frontier_anchors,
            ["learning", "owner", "paraphrases"],
            include_frontier_anchors=True,
            prioritize_remaining_profile_binding=True,
        )

        self.assertEqual(setup.profile_anchor_pool, repair_anchors + frontier_anchors)
        self.assertEqual(setup.frontier_targets_by_profile["qa:learning"], {5})
        self.assertEqual(setup.frontier_targets_by_profile["fact:owner"], {4})
        self.assertEqual(
            [profile for profile, _anchors in setup.profile_items],
            ["qa:learning", "fact:owner", "fact:self"],
        )
        self.assertEqual(setup.remaining_source_profiles, ["qa:learning"])

    def test_baseline_floor_profile_attempt_records_guard_counters(self) -> None:
        guard = {
            "sequential_profile_attempts": 0,
            "profile_scale_memory_attempts": 0,
            "profile_scale_frontier_attempts": 0,
            "profile_scale_frontier_records": 0,
            "profile_scale_coverage_frontier_attempts": 0,
            "profile_scale_coverage_prep_frontier_attempts": 0,
            "profile_scale_diversity_attempts": 0,
            "profile_scale_remaining_profile_binding_prioritized_attempts": 0,
            "profile_scale_owner_paraphrase_binding_prioritized_attempts": 0,
            "profile_scale_memory_consolidation_prioritized_attempts": 0,
            "sequential_profile_records": 0,
            "stabilization_anchor_batches": 0,
            "stabilization_anchor_records": 0,
        }

        profile_batch, frontier_records = baseline_floor_profile_attempt(
            profile="qa:learning",
            profile_anchors=[([0], 1, 1, "qa:learning")],
            rng=random.Random(7),
            frontier_targets_by_profile={"qa:learning": {1}},
            update_guard=guard,
            frontier_active=True,
            coverage_frontier_active=True,
            coverage_prep_frontier_active=True,
            diversity_active=True,
            remaining_profile_binding_prioritized=True,
            owner_paraphrase_binding_prioritized=True,
            memory_consolidation_prioritized=True,
        )

        self.assertEqual(profile_batch, [([0], 1, 1, "qa:learning")])
        self.assertEqual(frontier_records, 1)
        self.assertEqual(guard["sequential_profile_attempts"], 1)
        self.assertEqual(guard["profile_scale_memory_attempts"], 1)
        self.assertEqual(guard["profile_scale_frontier_attempts"], 1)
        self.assertEqual(guard["profile_scale_frontier_records"], 1)
        self.assertEqual(guard["profile_scale_coverage_frontier_attempts"], 1)
        self.assertEqual(guard["profile_scale_coverage_prep_frontier_attempts"], 1)
        self.assertEqual(guard["profile_scale_diversity_attempts"], 1)
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_prioritized_attempts"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_prioritized_attempts"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_memory_consolidation_prioritized_attempts"],
            1,
        )
        self.assertEqual(guard["sequential_profile_records"], 1)
        self.assertEqual(guard["stabilization_anchor_batches"], 1)
        self.assertEqual(guard["stabilization_anchor_records"], 1)

    def test_baseline_floor_objective_prompt_mode_records_objective_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-objective-screen"),
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
                        "branch-balanced-context-profile-baseline-floor-objective-"
                        "prompt-ownership-target-share-preserving-deficit-"
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
        self.assertTrue(direct_answer["direct_answer_replay_prediction_anchors_active"])
        self.assertTrue(direct_answer["direct_answer_baseline_floor_update_gate_active"])
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_adaptive_updates_active"]
        )
        self.assertTrue(direct_answer["direct_answer_baseline_floor_objective_active"])
        self.assertTrue(guard["active"])
        self.assertTrue(guard["adaptive"])
        self.assertTrue(guard["objective_active"])
        self.assertEqual(guard["objective_anchor_batch_size"], 32)
        self.assertEqual(guard["objective_anchor_weight"], 10.0)
        self.assertEqual(
            guard["objective_anchor_count"],
            replay_plan["baseline_floor_objective_anchor_count"],
        )
        self.assertEqual(
            replay_plan["baseline_floor_objective_anchor_batch_size"],
            32,
        )
        self.assertEqual(
            replay_plan["baseline_floor_objective_anchor_weight"],
            10.0,
        )
        self.assertEqual(guard["checked_steps"], 1)
        self.assertGreaterEqual(guard["attempted_updates"], guard["checked_steps"])
        self.assertEqual(
            guard["accepted_steps"] + guard["rejected_steps"],
            guard["checked_steps"],
        )
        self.assertEqual(
            guard["accepted_attempts"] + guard["rejected_attempts"],
            guard["attempted_updates"],
        )

    def test_baseline_floor_stabilization_mode_records_stabilization_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-stabilization-screen"),
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
                    "branch-context-profile-baseline-floor-stabilization-unlikelihood",
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
        self.assertTrue(direct_answer["direct_answer_replay_prediction_anchors_active"])
        self.assertTrue(direct_answer["direct_answer_baseline_floor_update_gate_active"])
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_adaptive_updates_active"]
        )
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_stabilization_active"]
        )
        self.assertTrue(guard["active"])
        self.assertTrue(guard["adaptive"])
        self.assertTrue(guard["stabilization_active"])
        self.assertEqual(guard["stabilization_anchor_batch_size"], 32)
        self.assertEqual(
            guard["stabilization_anchor_count"],
            replay_plan["baseline_floor_stabilization_anchor_count"],
        )
        self.assertEqual(
            replay_plan["baseline_floor_stabilization_anchor_batch_size"],
            32,
        )
        self.assertEqual(guard["checked_steps"], 1)
        self.assertGreaterEqual(guard["attempted_updates"], guard["checked_steps"])
        self.assertEqual(
            guard["accepted_steps"] + guard["rejected_steps"],
            guard["checked_steps"],
        )
        self.assertEqual(
            guard["accepted_attempts"] + guard["rejected_attempts"],
            guard["attempted_updates"],
        )
        self.assertTrue(guard["floor_diagnostics_active"])
        self.assertIn("rejected_update_shape_counts", guard)
        self.assertIn("rejected_learning_rate_scale_counts", guard)
        self.assertIn("rejected_violation_profile_counts", guard)
        self.assertIn("rejected_floor_diagnostic_sample", guard)
        self.assertGreaterEqual(guard["worst_rejected_coverage_deficit"], 0.0)
        if guard["rejected_attempts"]:
            self.assertEqual(
                sum(guard["rejected_update_shape_counts"].values()),
                guard["rejected_attempts"],
            )
            self.assertEqual(
                sum(guard["rejected_learning_rate_scale_counts"].values()),
                guard["rejected_attempts"],
            )
            self.assertIn("stabilization", guard["rejected_update_shape_counts"])
            self.assertTrue(guard["rejected_floor_diagnostic_sample"])
            self.assertIn(
                "worst_violation",
                guard["rejected_floor_diagnostic_sample"][0],
            )

    def test_profile_targeted_stabilization_mode_records_full_floor_surface(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-profile-targeted-screen"),
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
                        "branch-context-profile-baseline-floor-profile-targeted-"
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
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-sequential-screen"),
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
                        "branch-context-profile-baseline-floor-sequential-profile-"
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
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-calibrated-screen"),
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
                        "branch-context-profile-baseline-floor-calibrated-"
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

    def test_profile_scale_calibrated_stabilization_mode_records_scale_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-profile-scale-screen"),
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
                        "branch-context-profile-baseline-floor-profile-scale-"
                        "calibrated-sequential-profile-stabilization-unlikelihood"
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
                "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_calibrated_stabilization_active"])
        self.assertEqual(guard["outer_learning_rate_scales"], [1.0])
        self.assertEqual(replay_plan["outer_learning_rate_scales"], [1.0])
        self.assertIn(0.0001, guard["learning_rate_scales"])
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_calibrated_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_memory_attempts"],
            guard["sequential_profile_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_memory_acceptances"]
            + guard["profile_scale_memory_rejections"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertIn("profile_scale_probe_sample", guard)
        if guard["profile_scale_memory_attempts"]:
            self.assertTrue(guard["profile_scale_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_diversity_stabilization_mode_records_diversity_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-profile-scale-diversity-screen"),
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
                        "branch-context-profile-baseline-floor-diversity-profile-"
                        "scale-calibrated-sequential-profile-stabilization-"
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
                "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_diversity_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_calibrated_stabilization_active"])
        self.assertTrue(guard["profile_scale_diversity_stabilization_active"])
        self.assertEqual(guard["outer_learning_rate_scales"], [1.0])
        self.assertEqual(replay_plan["outer_learning_rate_scales"], [1.0])
        self.assertEqual(
            replay_plan["baseline_floor_profile_scale_diversity_stabilization_active"],
            True,
        )
        self.assertEqual(
            guard["profile_scale_diversity_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_diversity_acceptances"]
            + guard["profile_scale_diversity_rejections"],
            guard["profile_scale_diversity_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_diversity_score_improvements"]
            + guard["profile_scale_diversity_score_ties"]
            + guard["profile_scale_diversity_score_regressions"]
            + guard["profile_scale_diversity_floor_rejections"],
            guard["profile_scale_diversity_attempts"],
        )
        self.assertIn("profile_scale_diversity_probe_sample", guard)
        if guard["profile_scale_diversity_attempts"]:
            self.assertTrue(guard["profile_scale_diversity_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_frontier_stabilization_mode_records_frontier_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-profile-scale-frontier-screen"),
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
                        "frontier-profile-scale-calibrated-sequential-profile-"
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
                "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_diversity_stabilization_active"
            ]
        )
        self.assertTrue(
            direct_answer[
                "direct_answer_baseline_floor_profile_scale_frontier_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_calibrated_stabilization_active"])
        self.assertTrue(guard["profile_scale_diversity_stabilization_active"])
        self.assertTrue(guard["profile_scale_frontier_stabilization_active"])
        self.assertEqual(
            replay_plan["baseline_floor_profile_scale_frontier_stabilization_active"],
            True,
        )
        self.assertEqual(
            guard["frontier_anchor_count"],
            replay_plan["baseline_floor_frontier_anchor_count"],
        )
        self.assertGreaterEqual(guard["frontier_anchor_count"], 0)
        self.assertEqual(
            guard["profile_scale_frontier_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_frontier_acceptances"]
            + guard["profile_scale_frontier_rejections"],
            guard["profile_scale_frontier_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_diversity_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertIn("profile_scale_frontier_probe_sample", guard)
        if guard["profile_scale_frontier_attempts"]:
            self.assertTrue(guard["profile_scale_frontier_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_coverage_frontier_mode_records_coverage_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-profile-scale-coverage-frontier-screen"),
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
                        "coverage-frontier-profile-scale-calibrated-sequential-"
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
                "direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active"
            ]
        )
        self.assertTrue(guard["profile_scale_coverage_frontier_stabilization_active"])
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_coverage_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_acceptances"]
            + guard["profile_scale_coverage_frontier_rejections"],
            guard["profile_scale_coverage_frontier_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_gains"]
            + guard["profile_scale_coverage_frontier_ties"]
            + guard["profile_scale_coverage_frontier_regressions"],
            guard["profile_scale_coverage_frontier_attempts"],
        )
        if guard["profile_scale_coverage_frontier_acceptances"]:
            self.assertTrue(
                guard[
                    "profile_scale_coverage_frontier_profile_acceptance_deltas"
                ]
            )
        if guard["profile_scale_coverage_frontier_rejections"]:
            self.assertTrue(
                guard["profile_scale_coverage_frontier_rejection_reasons"]
            )
        self.assertIn("profile_scale_coverage_frontier_probe_sample", guard)
        if guard["profile_scale_coverage_frontier_attempts"]:
            self.assertTrue(guard["profile_scale_coverage_frontier_probe_sample"])
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_coverage_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )

    def test_profile_scale_coverage_prep_frontier_mode_records_preparation_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(
                        Path(temp)
                        / "baseline-floor-profile-scale-coverage-prep-frontier-screen"
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
                        "coverage-prep-frontier-profile-scale-calibrated-"
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
                "direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            guard["profile_scale_coverage_prep_frontier_stabilization_active"]
        )
        self.assertEqual(
            replay_plan[
                "baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active"
            ],
            True,
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_attempts"],
            guard["profile_scale_memory_attempts"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_acceptances"]
            + guard["profile_scale_coverage_prep_frontier_rejections"],
            guard["profile_scale_coverage_prep_frontier_attempts"],
        )
        self.assertLessEqual(
            guard["profile_scale_coverage_prep_frontier_gain_acceptances"]
            + guard["profile_scale_coverage_prep_frontier_preparations"],
            guard["profile_scale_coverage_prep_frontier_acceptances"],
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_attempts"],
            guard["profile_scale_coverage_prep_frontier_attempts"],
        )
        self.assertIn("profile_scale_coverage_prep_frontier_probe_sample", guard)
        if guard["profile_scale_coverage_prep_frontier_attempts"]:
            self.assertTrue(
                guard["profile_scale_coverage_prep_frontier_probe_sample"]
            )
        if guard["profile_scale_coverage_prep_frontier_acceptances"]:
            self.assertTrue(
                guard[
                    "profile_scale_coverage_prep_frontier_profile_acceptance_outcomes"
                ]
            )
        if guard["profile_scale_coverage_prep_frontier_rejections"]:
            self.assertTrue(
                guard["profile_scale_coverage_prep_frontier_rejection_reasons"]
            )
        shape_counts = dict(guard["accepted_update_shape_counts"])
        shape_counts.update(guard["rejected_update_shape_counts"])
        self.assertIn(
            "profile_scale_coverage_prep_frontier_diversity_calibrated_sequential_profile_stabilization",
            shape_counts,
        )


if __name__ == "__main__":
    unittest.main()
