from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_registry import (
    ExperimentIntent,
    read_experiment_intent,
    record_experiment_decision,
    write_experiment_intent,
)


class ExperimentRegistryTest(unittest.TestCase):
    def intent(self) -> dict:
        return ExperimentIntent(
            version="v0.71",
            run_id="run-001",
            component="test-component",
            hypothesis="A declared run can be checked before promotion.",
            allowed_data_sources=["corpus/train.txt"],
            planned_artifacts=["runs/run-001/metrics.json"],
            training_recipe_id="test-recipe",
            acceptance_gates=[
                {
                    "name": "baseline_snapshot_recorded",
                    "rule": "A baseline must be present.",
                    "required": True,
                }
            ],
            failure_criteria=["The baseline is missing."],
        ).to_record()

    def test_intent_record_round_trips(self) -> None:
        record = self.intent()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "experiment_intent.json"

            write_experiment_intent(path, record)
            loaded = read_experiment_intent(path)

        self.assertEqual(loaded["kind"], "experiment_intent")
        self.assertEqual(loaded["decision"]["status"], "planned")
        self.assertFalse(loaded["decision"]["promoted"])

    def test_decision_updates_promoted_flag(self) -> None:
        promoted = record_experiment_decision(
            self.intent(),
            "promoted",
            "All gates passed.",
            [{"name": "baseline_snapshot_recorded", "passed": True}],
        )
        rejected = record_experiment_decision(
            self.intent(),
            "rejected",
            "A required gate failed.",
            [{"name": "baseline_snapshot_recorded", "passed": False}],
        )

        self.assertTrue(promoted["decision"]["promoted"])
        self.assertFalse(rejected["decision"]["promoted"])

    def test_duplicate_gate_names_are_rejected(self) -> None:
        record = self.intent()
        record["acceptance_gates"].append(dict(record["acceptance_gates"][0]))

        with self.assertRaises(ValueError):
            write_experiment_intent(Path("/tmp/not-written.json"), record)


if __name__ == "__main__":
    unittest.main()
