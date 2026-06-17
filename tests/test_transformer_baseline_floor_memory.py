import random
import unittest

from transformer_baseline_floor_memory import (
    try_baseline_floor_missing_first_token_consolidation,
)


def memory_guard() -> dict[str, object]:
    return {
        "profile_scale_memory_consolidation_missing_first_token_candidates": 0,
        "profile_scale_memory_consolidation_missing_first_token_attempts": 0,
        "profile_scale_memory_consolidation_missing_first_token_records": 0,
        "profile_scale_memory_consolidation_missing_first_token_acceptances": 0,
        "profile_scale_memory_consolidation_missing_first_token_fallback_acceptances": 0,
        "profile_scale_memory_consolidation_missing_first_token_rejections": 0,
        "profile_scale_memory_consolidation_missing_first_token_rejection_reasons": {},
    }


class FakeModel:
    def to_dict(self, _tokenizer: object) -> dict[str, object]:
        return {"model": "candidate"}


class FakeOptimizer:
    def to_dict(self) -> dict[str, object]:
        return {"optimizer": "candidate"}


class FakeSnapshotRecorder:
    def __init__(self) -> None:
        self.metadata: list[dict[str, object]] = []

    def record(
        self,
        _step: int,
        _loss: object,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        self.metadata.append(metadata)
        return {"snapshot": len(self.metadata)}


class TransformerBaselineFloorMemoryTest(unittest.TestCase):
    def test_missing_first_token_consolidation_accepts_target_gain(self) -> None:
        guard = memory_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []
        train_calls: list[float] = []

        def train_branch_diversity(
            _model: object,
            _batch: list[object],
            learning_rate: float,
            *_args: object,
            **_kwargs: object,
        ) -> float:
            train_calls.append(learning_rate)
            return 0.7

        def target_delta(
            _snapshot: dict[str, object],
            baseline: dict[str, object],
        ) -> dict[str, int]:
            if baseline.get("snapshot") == "base":
                return {"regressed_profile_count": 0, "improved_profile_count": 2}
            return {"regressed_profile_count": 0, "improved_profile_count": 0}

        result = try_baseline_floor_missing_first_token_consolidation(
            active=True,
            memory_consolidation_prioritized=True,
            floor_preserved=True,
            diversity_accepted=True,
            coverage_accepted=True,
            profile_score=(1.0,),
            profile_probe_snapshot={"snapshot": "prepared"},
            coverage_delta={"regressed_profile_count": 0},
            coverage_outcome="tied",
            coverage_rejection_reason="coverage_tie",
            profile_base_snapshot={"snapshot": "base"},
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            profile_batch=[([1], 2, 2, "qa:owner")],
            rng=random.Random(1),
            base_learning_rate=0.2,
            profile_scale=0.5,
            negative_weight=0.1,
            positive_weight=0.2,
            contrast_weight=0.3,
            params=["p"],
            direct_step=4,
            direct_baseline={"snapshot": "baseline"},
            snapshot_recorder=recorder,
            update_guard=guard,
            update_shape="profile_scale",
            profile="qa:owner",
            profile_frontier_records=1,
            target_profiles=("qa:target",),
            missing_first_token_ids_by_profile={"qa:target": [2]},
            profile_specific=False,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restore_calls.append((model_payload, optimizer_payload))
            ),
            diversity_outcome="tied",
            diversity_rejection_reason="",
            learning_rate_scales=(0.25,),
            select_anchor_batch=lambda anchors, _ids, _rng, _size: anchors,
            train_branch_diversity=train_branch_diversity,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (2.0,),
            target_coverage_delta=target_delta,
            profile_diversity_delta=lambda _snapshot, _baseline, _profiles: {
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
                "profiles": [{"coverage_delta": 0.5}],
            },
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.loss_total, 0.7)
        self.assertEqual(result.loss_count, 1)
        self.assertEqual(result.learning_rate_scale, 0.25)
        self.assertEqual(result.target_profiles, ["qa:target"])
        self.assertEqual(result.target_ids, [2])
        self.assertEqual(result.profile_score, (2.0,))
        self.assertEqual(result.diversity_outcome, "improved")
        self.assertEqual(result.coverage_outcome, "gained")
        self.assertEqual(result.coverage_rejection_reason, "")
        self.assertEqual(result.coverage_delta["improved_profile_count"], 2)
        self.assertEqual(len(restore_calls), 1)
        self.assertEqual(train_calls, [0.025])
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_acceptances"
            ],
            1,
        )
        self.assertEqual(
            recorder.metadata[0][
                "baseline_floor_profile_scale_memory_consolidation_missing_first_token_probe"
            ],
            True,
        )
        self.assertEqual(
            recorder.metadata[0]["missing_first_token_target_ids"],
            [2],
        )

    def test_missing_first_token_consolidation_restores_after_rejection(self) -> None:
        guard = memory_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []

        result = try_baseline_floor_missing_first_token_consolidation(
            active=True,
            memory_consolidation_prioritized=True,
            floor_preserved=True,
            diversity_accepted=True,
            coverage_accepted=True,
            profile_score=(1.0,),
            profile_probe_snapshot={"snapshot": "prepared"},
            coverage_delta={"regressed_profile_count": 0},
            coverage_outcome="tied",
            coverage_rejection_reason="coverage_tie",
            profile_base_snapshot={"snapshot": "base"},
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            profile_batch=[([1], 2, 2, "qa:owner")],
            rng=random.Random(1),
            base_learning_rate=0.2,
            profile_scale=0.5,
            negative_weight=0.1,
            positive_weight=0.2,
            contrast_weight=0.3,
            params=["p"],
            direct_step=4,
            direct_baseline={"snapshot": "baseline"},
            snapshot_recorder=recorder,
            update_guard=guard,
            update_shape="profile_scale",
            profile="qa:owner",
            profile_frontier_records=1,
            target_profiles=("qa:target",),
            missing_first_token_ids_by_profile={"qa:target": [2]},
            profile_specific=False,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restore_calls.append((model_payload, optimizer_payload))
            ),
            diversity_outcome="tied",
            diversity_rejection_reason="",
            learning_rate_scales=(0.25,),
            select_anchor_batch=lambda anchors, _ids, _rng, _size: anchors,
            train_branch_diversity=lambda *_args, **_kwargs: 0.7,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (2.0,),
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_diversity_delta=lambda _snapshot, _baseline, _profiles: {
                "regressed_profile_count": 1,
                "improved_profile_count": 0,
                "profiles": [{"coverage_delta": -0.5}],
            },
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.attempted)
        self.assertEqual(result.outcome, "target_profile_regressed")
        self.assertEqual(result.rejection_reason, "target_profile_regression")
        self.assertEqual(result.target_profiles, ["qa:target"])
        self.assertEqual(result.target_ids, [2])
        self.assertEqual(len(restore_calls), 2)
        self.assertEqual(
            guard["profile_scale_memory_consolidation_missing_first_token_rejections"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_rejection_reasons"
            ],
            {"target_profile_regression": 1},
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_fallback_acceptances"
            ],
            1,
        )
        self.assertEqual(
            recorder.metadata[0][
                "baseline_floor_profile_scale_memory_consolidation_profile_specific_missing_first_token_probe"
            ],
            False,
        )


if __name__ == "__main__":
    unittest.main()
