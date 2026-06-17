import unittest

from support.baseline_floor_recovery import (
    FakeModel,
    FakeOptimizer,
    FakeSnapshotRecorder,
    branch_recovery_guard,
)
from transformer_baseline_floor_branch_diversity_recovery import (
    try_baseline_floor_branch_diversity_recovery,
)


class TransformerBaselineFloorBranchDiversityRecoveryTest(unittest.TestCase):
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
