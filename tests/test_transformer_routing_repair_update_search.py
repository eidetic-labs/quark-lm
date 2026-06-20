from __future__ import annotations

import unittest

from support.routing_repair_update_search import (
    RoutingRepairRecorder,
    routing_repair_context,
    routing_repair_guard,
    routing_repair_snapshot,
)
from transformer_routing_repair_update_search import (
    ROUTING_REPAIR_LEARNING_RATE_SCALES,
    apply_routing_repair_update_search,
)


class TransformerRoutingRepairUpdateSearchTests(unittest.TestCase):
    def test_accepts_first_coverage_preserving_retry_scale(self) -> None:
        recorder = RoutingRepairRecorder(
            [routing_repair_snapshot(0.25), routing_repair_snapshot(0.75)]
        )
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = routing_repair_guard()

        result = apply_routing_repair_update_search(
            routing_repair_context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(train_rates, [0.04, 0.02])
        self.assertEqual(len(restores), 2)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], 2)
        self.assertEqual(guard["rejected_attempts"], 1)
        self.assertEqual(guard["accepted_steps"], 1)
        self.assertEqual(guard["rejected_steps"], 0)
        self.assertEqual(guard["routing_repair_accepted_learning_rate_scale"], 0.5)
        self.assertEqual(guard["routing_repair_optimizer_probe_count"], 2)
        self.assertEqual(guard["routing_repair_optimizer_update_applied_count"], 2)
        self.assertEqual(guard["routing_repair_optimizer_nonzero_gradient_count"], 2)

    def test_rejects_preserved_coverage_without_response(self) -> None:
        recorder = RoutingRepairRecorder(
            [routing_repair_snapshot(0.5)] * len(ROUTING_REPAIR_LEARNING_RATE_SCALES)
        )
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = routing_repair_guard()

        result = apply_routing_repair_update_search(
            routing_repair_context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(guard["accepted_steps"], 0)
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertEqual(
            guard["routing_repair_no_response_rejections"],
            len(ROUTING_REPAIR_LEARNING_RATE_SCALES),
        )
        self.assertIsNone(guard["routing_repair_accepted_learning_rate_scale"])

    def test_accepts_preserved_coverage_with_rank_response(self) -> None:
        recorder = RoutingRepairRecorder(
            [routing_repair_snapshot(0.5, target_rank=8.0, top3_rate=0.25)]
        )
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = routing_repair_guard()

        result = apply_routing_repair_update_search(
            routing_repair_context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(guard["accepted_steps"], 1)
        self.assertEqual(guard["rejected_steps"], 0)
        self.assertEqual(guard["routing_repair_branch_response_acceptances"], 1)
        self.assertEqual(guard["routing_repair_accepted_learning_rate_scale"], 1.0)

    def test_rejects_all_scales_and_restores_baseline(self) -> None:
        recorder = RoutingRepairRecorder(
            [routing_repair_snapshot(0.25)]
            * len(ROUTING_REPAIR_LEARNING_RATE_SCALES)
        )
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = routing_repair_guard()

        result = apply_routing_repair_update_search(
            routing_repair_context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(len(train_rates), len(ROUTING_REPAIR_LEARNING_RATE_SCALES))
        self.assertEqual(len(restores), len(ROUTING_REPAIR_LEARNING_RATE_SCALES) + 1)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(
            guard["attempted_updates"],
            len(ROUTING_REPAIR_LEARNING_RATE_SCALES),
        )
        self.assertEqual(
            guard["rejected_attempts"],
            len(ROUTING_REPAIR_LEARNING_RATE_SCALES),
        )
        self.assertEqual(guard["accepted_steps"], 0)
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertIsNone(guard["routing_repair_accepted_learning_rate_scale"])


if __name__ == "__main__":
    unittest.main()
