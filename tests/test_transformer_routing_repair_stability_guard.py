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


class TransformerRoutingRepairStabilityGuardTests(unittest.TestCase):
    def test_rejects_branch_response_when_stability_regresses(self) -> None:
        unstable_response = routing_repair_snapshot(
            0.75,
            target_rank=8.0,
            top3_rate=0.25,
            predicted_unique=1,
            dominant_rate=1.0,
        )
        baseline = routing_repair_snapshot(
            0.5,
            predicted_unique=2,
            dominant_rate=0.5,
        )
        recorder = RoutingRepairRecorder(
            [unstable_response] * len(ROUTING_REPAIR_LEARNING_RATE_SCALES)
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
                baseline=baseline,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(guard["accepted_steps"], 0)
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertEqual(
            guard["routing_repair_stability_rejections"],
            len(ROUTING_REPAIR_LEARNING_RATE_SCALES),
        )
        self.assertEqual(
            guard["rejected_stability_violation_counts"]["newly_collapsed"],
            len(ROUTING_REPAIR_LEARNING_RATE_SCALES),
        )
        self.assertIsNone(guard["routing_repair_accepted_learning_rate_scale"])


if __name__ == "__main__":
    unittest.main()
