import unittest

from transformer_baseline_floor_branch_diversity_recovery import (
    try_baseline_floor_branch_diversity_recovery,
)
from transformer_baseline_floor_coverage_recovery import (
    try_baseline_floor_coverage_recovery,
)


def branch_recovery_guard() -> dict[str, object]:
    return {
        "profile_scale_branch_diversity_recovery_frontier_candidates": 0,
        "profile_scale_branch_diversity_recovery_frontier_attempts": 0,
        "profile_scale_branch_diversity_recovery_frontier_records": 0,
        "profile_scale_branch_diversity_recovery_frontier_acceptances": 0,
        "profile_scale_branch_diversity_recovery_frontier_fallback_acceptances": 0,
        "profile_scale_branch_diversity_recovery_frontier_rejections": 0,
        "profile_scale_branch_diversity_recovery_frontier_rejection_reasons": {},
    }


def coverage_recovery_guard() -> dict[str, object]:
    return {
        "profile_scale_coverage_recovery_frontier_prepared_candidates": 0,
        "profile_scale_coverage_recovery_frontier_attempts": 0,
        "profile_scale_coverage_recovery_frontier_records": 0,
        "profile_scale_coverage_recovery_frontier_acceptances": 0,
        "profile_scale_coverage_recovery_frontier_rejections": 0,
        "profile_scale_coverage_recovery_frontier_rejection_reasons": {},
        "profile_scale_branch_stable_coverage_recovery_frontier_checks": 0,
        "profile_scale_branch_stable_coverage_recovery_frontier_acceptances": 0,
        "profile_scale_branch_stable_coverage_recovery_frontier_rejections": 0,
        "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons": {},
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


class TransformerBaselineFloorRecoveryTest(unittest.TestCase):
    def test_coverage_recovery_accepts_gain(self) -> None:
        guard = coverage_recovery_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []
        train_calls: list[tuple[list[object], float]] = []

        def train_anchor_batch(
            _model: object,
            batch: list[object],
            learning_rate: float,
            **_kwargs: object,
        ) -> float:
            train_calls.append((batch, learning_rate))
            return 0.6

        result = try_baseline_floor_coverage_recovery(
            active=True,
            coverage_prep_accepted=True,
            profile_base_snapshot={"snapshot": "base"},
            profile_base_score=(1.0,),
            profile_score=(1.0,),
            profile_probe_snapshot={"snapshot": "prepared"},
            coverage_delta={"regressed_profile_count": 0},
            coverage_outcome="tied",
            coverage_rejection_reason="coverage_tie",
            floor_preserved=True,
            diversity_outcome="tied",
            diversity_rejection_reason="",
            branch_stable_active=True,
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            profile_batch=[
                ([1], 2, 2, "qa:owner"),
                ([1], 3, 3, "qa:owner"),
            ],
            frontier_targets_by_profile={"qa:owner": {2}},
            base_learning_rate=0.2,
            profile_scale=0.5,
            params=["p"],
            direct_step=4,
            direct_baseline={"snapshot": "baseline"},
            snapshot_recorder=recorder,
            update_guard=guard,
            update_shape="profile_scale",
            profile="qa:owner",
            profile_frontier_records=1,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restore_calls.append((model_payload, optimizer_payload))
            ),
            learning_rate_scales=(0.25,),
            train_anchor_batch=train_anchor_batch,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (2.0,),
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.loss_total, 0.6)
        self.assertEqual(result.loss_count, 1)
        self.assertEqual(result.learning_rate_scale, 0.25)
        self.assertEqual(result.records, 1)
        self.assertEqual(result.profile_score, (2.0,))
        self.assertEqual(result.diversity_outcome, "improved")
        self.assertEqual(result.coverage_outcome, "gained")
        self.assertEqual(result.coverage_rejection_reason, "")
        self.assertFalse(result.coverage_prep_accepted)
        self.assertTrue(result.branch_stable_checked)
        self.assertTrue(result.branch_stable_accepted)
        self.assertEqual(len(restore_calls), 1)
        self.assertEqual(len(train_calls[0][0]), 1)
        self.assertAlmostEqual(train_calls[0][1], 0.025)
        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_acceptances"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_acceptances"
            ],
            1,
        )
        self.assertEqual(
            recorder.metadata[0][
                "baseline_floor_profile_scale_coverage_recovery_probe"
            ],
            True,
        )

    def test_coverage_recovery_restores_after_rejection(self) -> None:
        guard = coverage_recovery_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []

        result = try_baseline_floor_coverage_recovery(
            active=True,
            coverage_prep_accepted=True,
            profile_base_snapshot={"snapshot": "base"},
            profile_base_score=(1.0,),
            profile_score=(1.0,),
            profile_probe_snapshot={"snapshot": "prepared"},
            coverage_delta={"regressed_profile_count": 0},
            coverage_outcome="tied",
            coverage_rejection_reason="coverage_tie",
            floor_preserved=True,
            diversity_outcome="tied",
            diversity_rejection_reason="",
            branch_stable_active=False,
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            profile_batch=[([1], 2, 2, "qa:owner")],
            frontier_targets_by_profile={},
            base_learning_rate=0.2,
            profile_scale=0.5,
            params=["p"],
            direct_step=4,
            direct_baseline={"snapshot": "baseline"},
            snapshot_recorder=recorder,
            update_guard=guard,
            update_shape="profile_scale",
            profile="qa:owner",
            profile_frontier_records=1,
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restore_calls.append((model_payload, optimizer_payload))
            ),
            learning_rate_scales=(0.25,),
            train_anchor_batch=lambda *_args, **_kwargs: 0.6,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (1.0,),
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.attempted)
        self.assertEqual(result.outcome, "coverage_tied")
        self.assertEqual(result.rejection_reason, "coverage_tie")
        self.assertTrue(result.coverage_prep_accepted)
        self.assertEqual(result.coverage_outcome, "tied")
        self.assertEqual(result.coverage_rejection_reason, "coverage_tie")
        self.assertEqual(len(restore_calls), 2)
        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_rejections"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_rejection_reasons"],
            {"coverage_tie": 1},
        )

    def test_branch_diversity_recovery_accepts_improved_candidate(self) -> None:
        guard = branch_recovery_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []

        result = try_baseline_floor_branch_diversity_recovery(
            active=True,
            floor_preserved=True,
            diversity_accepted=True,
            coverage_accepted=True,
            profile_score=(1.0,),
            profile_probe_snapshot={"snapshot": "prepared"},
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            profile_batch=[([1], 2, 2, "qa:owner")],
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
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restore_calls.append((model_payload, optimizer_payload))
            ),
            diversity_outcome="tied",
            diversity_rejection_reason="",
            learning_rate_scales=(0.25,),
            train_branch_diversity=lambda *_args, **_kwargs: 0.75,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (2.0,),
            target_coverage_delta=lambda _snapshot, _prepared: {
                "regressed_profile_count": 0,
                "improved_profile_count": 1,
            },
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.loss_total, 0.75)
        self.assertEqual(result.loss_count, 1)
        self.assertEqual(result.learning_rate_scale, 0.25)
        self.assertEqual(result.profile_score, (2.0,))
        self.assertEqual(result.diversity_outcome, "improved")
        self.assertEqual(result.diversity_rejection_reason, "")
        self.assertEqual(len(restore_calls), 1)
        self.assertEqual(
            guard["profile_scale_branch_diversity_recovery_frontier_acceptances"],
            1,
        )
        self.assertEqual(
            recorder.metadata[0][
                "baseline_floor_profile_scale_branch_diversity_recovery_probe"
            ],
            True,
        )

    def test_branch_diversity_recovery_restores_after_rejection(self) -> None:
        guard = branch_recovery_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []

        result = try_baseline_floor_branch_diversity_recovery(
            active=True,
            floor_preserved=True,
            diversity_accepted=True,
            coverage_accepted=True,
            profile_score=(1.0,),
            profile_probe_snapshot={"snapshot": "prepared"},
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            profile_batch=[([1], 2, 2, "qa:owner")],
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
            restore_direct_update_state=lambda model_payload, optimizer_payload: (
                restore_calls.append((model_payload, optimizer_payload))
            ),
            diversity_outcome="tied",
            diversity_rejection_reason="",
            learning_rate_scales=(0.25,),
            train_branch_diversity=lambda *_args, **_kwargs: 0.75,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (1.0,),
            target_coverage_delta=lambda _snapshot, _prepared: {
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.attempted)
        self.assertEqual(result.outcome, "score_tied")
        self.assertEqual(result.rejection_reason, "score_tie")
        self.assertEqual(result.profile_score, (1.0,))
        self.assertEqual(len(restore_calls), 2)
        self.assertEqual(
            guard["profile_scale_branch_diversity_recovery_frontier_rejections"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_diversity_recovery_frontier_rejection_reasons"
            ],
            {"score_tie": 1},
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_diversity_recovery_frontier_fallback_acceptances"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
