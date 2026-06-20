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
    ROUTING_REPAIR_BATCH_MODE,
    ROUTING_REPAIR_RANK_BATCH_MODE,
    ROUTING_REPAIR_RETENTION_RANK_BATCH_MODE,
    ROUTING_REPAIR_TOPK_BATCH_MODE,
    record_routing_repair_batch_step,
    routing_repair_batch_evidence_summary,
)
from transformer_routing_repair_bundle import (  # noqa: E402
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
)


class TransformerRoutingRepairBatchEvidenceTests(unittest.TestCase):
    def test_records_profile_balanced_training_family_batch(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = _args()

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
        self.assertEqual(record["step"], 1)
        self.assertEqual(record["profiles"], ["owner", "qa"])
        self.assertEqual(record["branch_count"], 2)

    def test_records_profile_balanced_rank_bundle_batch(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = _args(
            mode=ROUTING_REPAIR_RANK_BATCH_MODE,
            bundle=PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
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
        summary = routing_repair_batch_evidence_summary(
            args,
            [record] if record is not None else [],
            _baseline(),
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["bundle"], PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE)
        self.assertEqual(summary["direct_answer_mode"], ROUTING_REPAIR_RANK_BATCH_MODE)

    def test_records_profile_balanced_topk_bundle_batch(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = _args(
            mode=ROUTING_REPAIR_TOPK_BATCH_MODE,
            bundle=PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
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
        summary = routing_repair_batch_evidence_summary(
            args,
            [record] if record is not None else [],
            _baseline(),
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["bundle"], PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE)
        self.assertEqual(summary["direct_answer_mode"], ROUTING_REPAIR_TOPK_BATCH_MODE)

    def test_records_retention_anchors_for_retention_rank_bundle(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = _args(
            mode=ROUTING_REPAIR_RETENTION_RANK_BATCH_MODE,
            bundle=PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
        )
        represented_target = fixture.tokenizer.stoi[fixture.near.target[1]]

        def predict_represented_target(_context: list[int]) -> list[float]:
            probs = [0.0 for _token in fixture.tokenizer.tokens]
            probs[represented_target] = 1.0
            return probs

        fixture.model.predict = predict_represented_target

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
        self.assertGreater(record["retention_anchor_count"], 0)
        self.assertTrue(record["retention_anchor_profile_counts"])

        summary = routing_repair_batch_evidence_summary(
            args,
            [record],
            _baseline(),
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertGreater(summary["retention_anchor_count"], 0)

    def test_summary_covers_trainable_failed_profiles(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = _args()
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
        self.assertEqual(check["covered_trainable_profiles"], ["owner", "qa"])
        self.assertEqual(check["eval_only_profiles"], ["heldout"])

    def test_wrong_mode_does_not_record_bundle_evidence(self) -> None:
        fixture = branch_training_fixture(seed=40)
        args = _args(mode="branch-target-margin-unlikelihood")

        record = record_routing_repair_batch_step(
            args=args,
            model=fixture.model,
            tokenizer=fixture.tokenizer,
            branch_examples=fixture.examples,
            rng=random.Random(11),
            direct_step=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertIsNone(record)
        self.assertIsNone(routing_repair_batch_evidence_summary(args, [], _baseline()))


def _args(
    mode: str = ROUTING_REPAIR_BATCH_MODE,
    bundle: str = PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
) -> Namespace:
    return Namespace(
        experiment_bundle=bundle,
        direct_answer_mode=mode,
        direct_answer_branch_position=1,
        direct_answer_branch_batch_size=2,
    )


def _baseline() -> dict:
    return {
        "branch_diversity_target": {
            "root_cause": {
                "profiles": [
                    {
                        "name": "heldout",
                        "failure_modes": ["targets_buried"],
                    },
                    {
                        "name": "owner",
                        "failure_modes": ["target_coverage_gap"],
                    },
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
