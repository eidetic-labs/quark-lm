from __future__ import annotations

import random
import tempfile
import unittest

from support.baseline_floor_modes import train_baseline_floor_mode_screen
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
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-objective-screen",
                (
                    "branch-balanced-context-profile-baseline-floor-objective-"
                    "prompt-ownership-target-share-preserving-deficit-"
                    "unlikelihood"
                ),
            )

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
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-stabilization-screen",
                "branch-context-profile-baseline-floor-stabilization-unlikelihood",
            )

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


if __name__ == "__main__":
    unittest.main()
