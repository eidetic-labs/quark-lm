from __future__ import annotations

import random
import unittest
from argparse import Namespace

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from transformer_routing_repair_batch_evidence import (
    ROUTING_REPAIR_BATCH_MODE,
    record_routing_repair_batch_step,
    routing_repair_batch_evidence_summary,
)
from transformer_routing_repair_bundle import PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE


class TransformerRoutingRepairTargetProfilesTest(unittest.TestCase):
    def test_summary_records_declared_repair_target_profiles(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = _args(["owner"])
        record = record_routing_repair_batch_step(
            args=args,
            model=fixture.model,
            tokenizer=fixture.tokenizer,
            branch_examples=fixture.examples,
            rng=random.Random(11),
            direct_step=1,
            terminator=ANSWER_TERMINATOR,
        )

        summary = routing_repair_batch_evidence_summary(
            args,
            [record] if record is not None else [],
            _baseline(),
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        check = summary["profile_balanced_branch_batches"]
        self.assertTrue(check["passed"])
        self.assertEqual(summary["declared_repair_target_profiles"], ["owner"])
        self.assertEqual(check["declared_repair_target_profiles"], ["owner"])
        self.assertEqual(check["all_failed_trainable_profiles"], ["owner", "qa"])
        self.assertEqual(check["required_trainable_profiles"], ["owner"])
        self.assertEqual(check["covered_trainable_profiles"], ["owner"])


def _args(repair_target_profiles: list[str]) -> Namespace:
    return Namespace(
        experiment_bundle=PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
        direct_answer_mode=ROUTING_REPAIR_BATCH_MODE,
        direct_answer_branch_position=1,
        direct_answer_branch_batch_size=2,
        direct_answer_repair_target_profile=repair_target_profiles,
    )


def _baseline() -> dict:
    return {
        "branch_diversity_target": {
            "root_cause": {
                "profiles": [
                    {"name": "owner", "failure_modes": ["target_coverage_gap"]},
                    {
                        "name": "qa",
                        "failure_modes": ["zero_target_coverage", "targets_buried"],
                    },
                ]
            }
        }
    }


if __name__ == "__main__":
    unittest.main()
