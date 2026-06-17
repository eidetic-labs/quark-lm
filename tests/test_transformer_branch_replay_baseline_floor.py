from __future__ import annotations

import random
import tempfile
import unittest

from support.baseline_floor import (
    baseline_floor_anchor_profile_groups,
    baseline_floor_anchor_profile_target_count,
    baseline_floor_objective_anchor_batch,
    baseline_floor_repair_anchor_records,
    train_direct_answer_baseline_floor_anchor_batch,
)
from support.baseline_floor_modes import train_baseline_floor_mode_screen
from support.core import CharTokenizer, TinyTransformerLM, TransformerConfig, context_before


class TransformerBranchReplayBaselineFloorTest(unittest.TestCase):
    def test_baseline_floor_gated_prompt_mode_records_update_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-gated-screen",
                (
                    "branch-balanced-context-profile-baseline-floor-gated-"
                    "prompt-ownership-target-share-preserving-deficit-"
                    "unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        self.assertTrue(direct_answer["direct_answer_replay_prediction_anchors_active"])
        self.assertTrue(direct_answer["direct_answer_baseline_floor_update_gate_active"])
        self.assertTrue(guard["active"])
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(
            guard["accepted_steps"] + guard["rejected_steps"],
            guard["checked_steps"],
        )

    def test_baseline_floor_adaptive_prompt_mode_records_retry_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-adaptive-screen",
                (
                    "branch-balanced-context-profile-baseline-floor-adaptive-"
                    "prompt-ownership-target-share-preserving-deficit-"
                    "unlikelihood"
                ),
            )

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        self.assertTrue(direct_answer["direct_answer_replay_prediction_anchors_active"])
        self.assertTrue(direct_answer["direct_answer_baseline_floor_update_gate_active"])
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_adaptive_updates_active"]
        )
        self.assertTrue(guard["active"])
        self.assertTrue(guard["adaptive"])
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

    def test_baseline_floor_repair_anchor_records_keep_covered_predictions(
        self,
    ) -> None:
        anchors = baseline_floor_repair_anchor_records(
            [
                ([1, 2], 4, 4, "qa:place"),
                ([1, 3], 5, 4, "qa:place"),
                ([1, 4], 6, 9, "qa:place"),
                ([2, 2], 7, 8, "qa:owner"),
                ([2, 3], 8, 8, "qa:owner"),
            ]
        )

        self.assertEqual(
            anchors,
            [
                ([1, 2], 4, 4, "qa:place"),
                ([1, 3], 4, 4, "qa:place"),
                ([2, 2], 8, 8, "qa:owner"),
                ([2, 3], 8, 8, "qa:owner"),
            ],
        )

    def test_baseline_floor_objective_anchor_batch_balances_profile_targets(
        self,
    ) -> None:
        anchors = [
            ([1, 2], 4, 4, "qa:place"),
            ([1, 3], 4, 4, "qa:place"),
            ([2, 2], 8, 8, "qa:owner"),
            ([2, 3], 9, 9, "qa:owner"),
            ([3, 3], 9, 9, "qa:owner"),
        ]

        batch = baseline_floor_objective_anchor_batch(
            anchors,
            random.Random(11),
            batch_size=10,
        )

        profile_targets = {
            (profile, target)
            for _context, target, _predicted, profile in batch
        }
        self.assertEqual(
            profile_targets,
            {("qa:place", 4), ("qa:owner", 8), ("qa:owner", 9)},
        )
        self.assertEqual(baseline_floor_anchor_profile_target_count(anchors), 3)
        self.assertEqual(
            {
                profile: len(group)
                for profile, group in baseline_floor_anchor_profile_groups(
                    anchors
                ).items()
            },
            {"qa:owner": 3, "qa:place": 2},
        )

    def test_baseline_floor_anchor_batch_update_lowers_anchor_nll(self) -> None:
        text = "where? near.\nwho? owner.\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=89,
        )
        model = TinyTransformerLM.init_random(config)
        target = ids[4]
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        before = model.nll(context, target)

        for _step in range(10):
            train_direct_answer_baseline_floor_anchor_batch(
                model,
                [(context, target, target, "qa:place")],
                learning_rate=0.05,
            )

        self.assertLess(model.nll(context, target), before)

    def test_baseline_floor_repaired_prompt_mode_records_repair_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            metrics = train_baseline_floor_mode_screen(
                temp,
                "baseline-floor-repaired-screen",
                (
                    "branch-balanced-context-profile-baseline-floor-repaired-"
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
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_repaired_updates_active"]
        )
        self.assertTrue(guard["active"])
        self.assertTrue(guard["adaptive"])
        self.assertTrue(guard["repair_active"])
        self.assertGreaterEqual(guard["repair_anchor_count"], 0)
        self.assertEqual(
            guard["repair_anchor_count"],
            replay_plan["baseline_floor_repair_anchor_count"],
        )
        self.assertEqual(guard["repair_steps_per_attempt"], 1)
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


if __name__ == "__main__":
    unittest.main()
