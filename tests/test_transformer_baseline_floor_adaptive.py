import random
import unittest
from typing import Any

from transformer_baseline_floor_adaptive import (
    train_adaptive_baseline_floor_update_stage,
)


def empty_guard() -> dict[str, Any]:
    return {
        "checked_steps": 0,
        "attempted_updates": 0,
        "profile_scale_diversity_outer_acceptances": 0,
        "profile_scale_diversity_outer_rejections": 0,
        "rejected_no_effective_update_attempts": 0,
        "repair_attempts": 0,
        "rejected_steps": 0,
    }


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
            "metadata": metadata,
            "repair": bool(metadata.get("baseline_floor_repair_probe")),
        }
        self.records.append(snapshot)
        return snapshot


class TransformerBaselineFloorAdaptiveTest(unittest.TestCase):
    def test_adaptive_stage_accepts_preserved_primary_update(self) -> None:
        guard = empty_guard()
        rng = random.Random(3)
        base_rng_state = rng.getstate()
        recorder = FakeSnapshotRecorder()
        restores: list[tuple[dict[str, Any], dict[str, Any]]] = []
        acceptances: list[tuple[float, str]] = []

        loss = train_adaptive_baseline_floor_update_stage(
            model=object(),
            rng=rng,
            example="example",
            lesson="lesson",
            direct_step=5,
            base_model_payload={"model": "base"},
            base_optimizer_payload={"optimizer": "base"},
            base_rng_state=base_rng_state,
            direct_answer_mode="mode",
            direct_answer_learning_rate=0.2,
            direct_answer_branch_batch_size=2,
            direct_answer_hard_negatives=1,
            update_guard=guard,
            direct_baseline={"baseline": True},
            snapshot_recorder=recorder,
            outer_learning_rate_scales=[0.5],
            repair_anchors=[],
            repaired_updates_active=False,
            stabilization_active=True,
            profile_scale_diversity_active=False,
            train_stabilization_update=lambda learning_rate, step: (
                learning_rate + step,
                True,
            ),
            train_baseline_anchored_prompt=lambda *_args: 0.0,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restores.append((model_payload, optimizer_payload))
            ),
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            update_shape_for_mode=lambda mode: f"{mode}-shape",
            record_acceptance=lambda _guard, scale, shape: (
                acceptances.append((scale, shape))
            ),
        )

        self.assertEqual(loss, 5.1)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], 1)
        self.assertEqual(restores, [({"model": "base"}, {"optimizer": "base"})])
        self.assertEqual(acceptances, [(0.5, "mode-shape")])
        self.assertEqual(recorder.records[0]["metadata"]["learning_rate_scale"], 0.5)

    def test_adaptive_stage_accepts_repair_after_diversity_rejection(self) -> None:
        guard = empty_guard()
        rng = random.Random(7)
        base_rng_state = rng.getstate()
        recorder = FakeSnapshotRecorder()
        acceptances: list[tuple[float, str]] = []
        rejections: list[tuple[float, str]] = []

        loss = train_adaptive_baseline_floor_update_stage(
            model="model",
            rng=rng,
            example="example",
            lesson="lesson",
            direct_step=2,
            base_model_payload={},
            base_optimizer_payload={},
            base_rng_state=base_rng_state,
            direct_answer_mode="mode",
            direct_answer_learning_rate=0.1,
            direct_answer_branch_batch_size=3,
            direct_answer_hard_negatives=2,
            update_guard=guard,
            direct_baseline={"kind": "baseline"},
            snapshot_recorder=recorder,
            outer_learning_rate_scales=[1.0],
            repair_anchors=[([0], 1, 1, "fact:self")],
            repaired_updates_active=True,
            stabilization_active=True,
            profile_scale_diversity_active=True,
            train_stabilization_update=lambda _learning_rate, _step: (3.0, True),
            train_baseline_anchored_prompt=lambda *_args: 0.0,
            restore_direct_update_state=lambda *_args: None,
            train_repair_stage=lambda *_args, **_kwargs: 5.0,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            update_shape_for_mode=lambda _mode: "primary",
            snapshot_score=lambda snapshot: (
                (2.0,)
                if snapshot.get("kind") == "baseline"
                else (3.0,)
                if snapshot.get("repair")
                else (1.0,)
            ),
            record_acceptance=lambda _guard, scale, shape: (
                acceptances.append((scale, shape))
            ),
            record_rejection_attempt=lambda _guard, _baseline, _step, _snapshot, scale, shape: (
                rejections.append((scale, shape))
            ),
        )

        self.assertEqual(loss, 4.0)
        self.assertEqual(guard["profile_scale_diversity_outer_rejections"], 1)
        self.assertEqual(guard["profile_scale_diversity_outer_acceptances"], 0)
        self.assertEqual(guard["repair_attempts"], 1)
        self.assertEqual(acceptances, [(1.0, "repaired")])
        self.assertEqual(rejections, [])
        self.assertTrue(recorder.records[1]["repair"])

    def test_adaptive_stage_records_rejection_and_restores_base_state(self) -> None:
        guard = empty_guard()
        rng = random.Random(11)
        base_rng_state = rng.getstate()
        recorder = FakeSnapshotRecorder()
        restores: list[tuple[dict[str, Any], dict[str, Any]]] = []
        rejections: list[tuple[int, float, str]] = []

        loss = train_adaptive_baseline_floor_update_stage(
            model=object(),
            rng=rng,
            example="example",
            lesson="lesson",
            direct_step=9,
            base_model_payload={"model": "base"},
            base_optimizer_payload={"optimizer": "base"},
            base_rng_state=base_rng_state,
            direct_answer_mode="mode",
            direct_answer_learning_rate=0.1,
            direct_answer_branch_batch_size=2,
            direct_answer_hard_negatives=0,
            update_guard=guard,
            direct_baseline={},
            snapshot_recorder=recorder,
            outer_learning_rate_scales=[1.0],
            repair_anchors=[],
            repaired_updates_active=False,
            stabilization_active=True,
            profile_scale_diversity_active=False,
            train_stabilization_update=lambda _learning_rate, _step: (7.0, False),
            train_baseline_anchored_prompt=lambda *_args: 0.0,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restores.append((model_payload, optimizer_payload))
            ),
            preserves_target_coverage=lambda _snapshot, _baseline: False,
            update_shape_for_mode=lambda _mode: "direct",
            record_rejection_attempt=lambda _guard, _baseline, step, _snapshot, scale, shape: (
                rejections.append((step, scale, shape))
            ),
        )

        self.assertEqual(loss, 7.0)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], 1)
        self.assertEqual(guard["rejected_no_effective_update_attempts"], 1)
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertEqual(
            restores,
            [
                ({"model": "base"}, {"optimizer": "base"}),
                ({"model": "base"}, {"optimizer": "base"}),
            ],
        )
        self.assertEqual(rejections, [(9, 1.0, "direct")])


if __name__ == "__main__":
    unittest.main()
