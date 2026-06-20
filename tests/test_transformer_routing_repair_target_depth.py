from __future__ import annotations

import random
import sys
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.branch_training import branch_training_fixture  # noqa: E402
from support.core import ANSWER_TERMINATOR  # noqa: E402
from transformer_routing_repair_batch_evidence import (  # noqa: E402
    record_routing_repair_batch_step,
    routing_repair_batch_evidence_summary,
)
from transformer_routing_repair_bundle import (  # noqa: E402
    PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_MODE,
)


class TransformerRoutingRepairTargetDepthTests(unittest.TestCase):
    def test_rank_collapse_batch_evidence_records_target_depth(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = Namespace(
            experiment_bundle=PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE,
            direct_answer_mode=PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_MODE,
            direct_answer_branch_position=1,
            direct_answer_branch_batch_size=2,
        )

        record = record_routing_repair_batch_step(
            args=args,
            model=fixture.model,
            tokenizer=fixture.tokenizer,
            branch_examples=fixture.examples,
            rng=random.Random(11),
            direct_step=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["min_targets_per_profile"], 2)
        self.assertEqual(record["branch_count"], 3)

        summary = routing_repair_batch_evidence_summary(args, [record], _baseline())
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["min_targets_per_profile"], 2)


def _baseline() -> dict:
    return {
        "branch_diversity_target": {
            "root_cause": {
                "profiles": [
                    {"name": "owner", "failure_modes": ["target_coverage_gap"]},
                    {"name": "qa", "failure_modes": ["zero_target_coverage"]},
                ]
            }
        }
    }


if __name__ == "__main__":
    unittest.main()
