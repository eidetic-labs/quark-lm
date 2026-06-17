import random
import unittest
from typing import Any

from transformer_baseline_floor_sequential import (
    train_baseline_floor_sequential_profile_stage,
)


def empty_guard() -> dict[str, Any]:
    return {
        "sequential_profile_attempts": 0,
        "sequential_profile_records": 0,
        "stabilization_anchor_batches": 0,
        "stabilization_anchor_records": 0,
        "sequential_profile_acceptances": 0,
        "sequential_profile_rejections": 0,
        "sequential_profile_acceptance_counts": {},
        "sequential_profile_rejection_counts": {},
        "sequential_profile_probe_sample": [],
    }


class FakeModel:
    def to_dict(self, tokenizer: object) -> dict[str, Any]:
        return {"tokenizer": tokenizer}


class FakeOptimizer:
    def to_dict(self) -> dict[str, Any]:
        return {"optimizer": True}


class FakeSnapshotRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        step: int,
        loss: object,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = {
            "step": step,
            "loss": loss,
            "profile": metadata["sequential_profile"],
            "metadata": metadata,
        }
        self.records.append(snapshot)
        return snapshot


class TransformerBaselineFloorSequentialTest(unittest.TestCase):
    def test_sequential_stage_records_acceptances(self) -> None:
        guard = empty_guard()
        recorder = FakeSnapshotRecorder()
        restores: list[tuple[dict[str, Any], dict[str, Any]]] = []
        anchors = [
            ([0], 1, 1, "fact:self"),
            ([1], 2, 2, "qa:learning"),
        ]

        loss, accepted = train_baseline_floor_sequential_profile_stage(
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            repair_anchors=anchors,
            rng=random.Random(3),
            update_learning_rate=0.05,
            base_learning_rate=0.1,
            update_guard=guard,
            direct_step=7,
            direct_baseline={},
            snapshot_recorder=recorder,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restores.append((model_payload, optimizer_payload))
            ),
            calibrated=True,
            train_anchor_batch=lambda *_args, **_kwargs: 2.0,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
        )

        self.assertEqual(loss, 2.0)
        self.assertTrue(accepted)
        self.assertEqual(restores, [])
        self.assertEqual(guard["sequential_profile_attempts"], 2)
        self.assertEqual(guard["stabilization_anchor_batches"], 2)
        self.assertEqual(guard["sequential_profile_acceptances"], 2)
        self.assertEqual(guard["sequential_profile_rejections"], 0)
        self.assertEqual(
            guard["sequential_profile_acceptance_counts"],
            {"fact:self": 1, "qa:learning": 1},
        )
        self.assertEqual(
            [record["metadata"]["learning_rate_scale"] for record in recorder.records],
            [0.5, 0.5],
        )
        self.assertEqual(
            {sample["accepted"] for sample in guard["sequential_profile_probe_sample"]},
            {True},
        )

    def test_sequential_stage_restores_rejected_profiles(self) -> None:
        guard = empty_guard()
        recorder = FakeSnapshotRecorder()
        restores: list[tuple[dict[str, Any], dict[str, Any]]] = []
        anchors = [
            ([0], 1, 1, "fact:self"),
            ([1], 2, 2, "qa:learning"),
        ]

        loss, accepted = train_baseline_floor_sequential_profile_stage(
            model=FakeModel(),
            tokenizer="tokenizer",
            optimizer=FakeOptimizer(),
            repair_anchors=anchors,
            rng=random.Random(5),
            update_learning_rate=0.1,
            base_learning_rate=0.1,
            update_guard=guard,
            direct_step=3,
            direct_baseline={},
            snapshot_recorder=recorder,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restores.append((model_payload, optimizer_payload))
            ),
            calibrated=False,
            train_anchor_batch=lambda *_args, **_kwargs: 4.0,
            preserves_target_coverage=lambda snapshot, _baseline: (
                snapshot["profile"] == "qa:learning"
            ),
            coverage_diagnostics=lambda snapshot, _baseline: {
                "worst_violation": {
                    "profile": snapshot["profile"],
                    "deficit": 1,
                },
                "violating_profile_count": 1,
            },
        )

        self.assertEqual(loss, 4.0)
        self.assertTrue(accepted)
        self.assertEqual(restores, [({"tokenizer": "tokenizer"}, {"optimizer": True})])
        self.assertEqual(guard["sequential_profile_acceptances"], 1)
        self.assertEqual(guard["sequential_profile_rejections"], 1)
        self.assertEqual(
            guard["sequential_profile_rejection_counts"],
            {"fact:self": 1},
        )
        rejected_sample = guard["sequential_profile_probe_sample"][0]
        self.assertFalse(rejected_sample["accepted"])
        self.assertEqual(
            rejected_sample["worst_violation"],
            {"profile": "fact:self", "deficit": 1},
        )


if __name__ == "__main__":
    unittest.main()
