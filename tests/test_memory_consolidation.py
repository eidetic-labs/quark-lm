from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_consolidation import (
    build_memory_consolidation_plan,
    write_memory_consolidation_plan,
)


def retrieval_report() -> dict:
    return {
        "kind": "retrieval_memory_report",
        "summary": {"record_count": 2, "exact_rate": 1.0},
        "evals": {
            "owner": {
                "count": 2,
                "exact": 2,
                "exact_rate": 1.0,
                "retrieved": 2,
                "records": [
                    {
                        "id": "owner-map",
                        "target": " ivy.",
                        "retrieved": True,
                        "memory_card_id": "corpus:story:ivy-map:owner-question",
                        "memory_card_source": "corpus:grammar:story_facts",
                    },
                    {
                        "id": "owner-pen",
                        "target": " omar.",
                        "retrieved": True,
                        "memory_card_id": "corpus:story:omar-pen:owner-question",
                        "memory_card_source": "corpus:grammar:story_facts",
                    },
                ],
            },
            "unknowns": {
                "count": 1,
                "exact": 1,
                "exact_rate": 1.0,
                "retrieved": 0,
                "records": [],
            },
        },
    }


def transformer_metrics() -> dict:
    return {
        "run_id": "test-run",
        "metrics_path": "runs/test-run/transformer_answer_metrics.json",
        "direct_answer": {
            "final": {
                "branch_diversity_target": {
                    "passed": False,
                    "failed_profiles": 2,
                    "blocking_evals": [
                        {
                            "name": "owner",
                            "collapsed": True,
                            "target_token_coverage": 0.125,
                            "predicted_unique": 1,
                            "target_unique": 8,
                            "dominant_predicted_token": "n",
                            "dominant_predicted_rate": 1.0,
                            "missing_target_tokens": [{"value": "i", "count": 1}],
                        },
                        {
                            "name": "unknowns",
                            "collapsed": False,
                            "target_token_coverage": 1.0,
                            "predicted_unique": 1,
                            "target_unique": 1,
                        },
                    ],
                }
            }
        },
    }


class MemoryConsolidationTest(unittest.TestCase):
    def test_plan_prioritizes_memory_backed_neural_failures(self) -> None:
        plan = build_memory_consolidation_plan(
            retrieval_report(),
            transformer_metrics(),
        )

        self.assertEqual(plan["kind"], "memory_consolidation_plan")
        self.assertFalse(plan["dataset_exclusivity"]["uses_external_model"])
        self.assertFalse(plan["dataset_exclusivity"]["updates_weights"])
        self.assertEqual(plan["summary"]["memory_backed_failed_profiles"], 1)
        self.assertEqual(plan["summary"]["collapsed_memory_backed_profiles"], ["owner"])
        self.assertEqual(plan["summary"]["top_priority_profiles"], ["owner"])
        owner = plan["profile_priorities"][0]
        self.assertEqual(owner["profile"], "owner")
        self.assertEqual(owner["retrieval_exact_rate"], 1.0)
        self.assertEqual(owner["retrieved_records"], 2)
        self.assertEqual(owner["neural_target_token_coverage"], 0.125)
        self.assertTrue(owner["collapsed"])
        self.assertEqual(len(owner["memory_cards"]), 2)
        self.assertEqual(
            owner["recommended_action"],
            "consolidate_retrieved_memory_with_branch_diversity_gate",
        )

    def test_plan_can_be_written_as_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "memory_consolidation_plan.json"
            plan = build_memory_consolidation_plan(
                retrieval_report(),
                transformer_metrics(),
            )

            write_memory_consolidation_plan(path, plan)

            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["summary"]["top_priority_profiles"], ["owner"])


if __name__ == "__main__":
    unittest.main()
