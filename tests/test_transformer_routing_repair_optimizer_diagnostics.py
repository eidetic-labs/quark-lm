from __future__ import annotations

import unittest
from types import SimpleNamespace

from transformer_routing_repair_optimizer_diagnostics import (
    record_routing_repair_optimizer_probe,
)


class TransformerRoutingRepairOptimizerDiagnosticsTest(unittest.TestCase):
    def test_records_compact_optimizer_probe_evidence(self) -> None:
        guard: dict = {}
        optimizer = SimpleNamespace(last_apply_evidence=_evidence())

        record_routing_repair_optimizer_probe(guard, optimizer, 0.5, 1.25)

        self.assertEqual(guard["routing_repair_optimizer_probe_count"], 1)
        self.assertEqual(guard["routing_repair_optimizer_update_applied_count"], 1)
        self.assertEqual(guard["routing_repair_optimizer_nonzero_gradient_count"], 1)
        sample = guard["routing_repair_optimizer_probe_sample"][0]
        self.assertEqual(sample["learning_rate_scale"], 0.5)
        self.assertEqual(sample["loss"], 1.25)
        self.assertEqual(sample["clipped_gradient_abs_sum"], 2.0)
        self.assertTrue(sample["accumulated_gradient_available"])

    def test_counts_missing_optimizer_evidence(self) -> None:
        guard: dict = {}

        record_routing_repair_optimizer_probe(guard, object(), 1.0, 0.0)

        self.assertEqual(guard["routing_repair_optimizer_probe_missing_evidence"], 1)


def _evidence() -> dict:
    return {
        "raw_gradient": {"signature": {"abs_sum": 3.0}},
        "clipped_gradient": {"signature": {"abs_sum": 2.0}},
        "accumulated_gradient": {
            "available": True,
            "signature": {"abs_sum": 1.0},
        },
        "update_applied": True,
        "learning_rate": 0.02,
        "update_count_before": 0,
        "update_count_after": 1,
        "pending_accumulation_before": 0,
        "pending_accumulation_after": 0,
    }


if __name__ == "__main__":
    unittest.main()
