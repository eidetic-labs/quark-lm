import unittest

from transformer_baseline_floor_binding import (
    try_baseline_floor_collapsed_profile_binding,
)


def binding_guard() -> dict[str, object]:
    return {
        "profile_scale_collapsed_profile_binding_frontier_candidates": 0,
        "profile_scale_collapsed_profile_binding_frontier_attempts": 0,
        "profile_scale_collapsed_profile_binding_frontier_records": 0,
        "profile_scale_collapsed_profile_binding_frontier_acceptances": 0,
        "profile_scale_collapsed_profile_binding_frontier_fallback_acceptances": 0,
        "profile_scale_collapsed_profile_binding_frontier_rejections": 0,
        "profile_scale_collapsed_profile_binding_frontier_rejection_reasons": {},
        "profile_scale_owner_paraphrase_binding_preservation_checks": 0,
        "profile_scale_owner_paraphrase_binding_preservation_failures": 0,
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


class TransformerBaselineFloorBindingTest(unittest.TestCase):
    def test_collapsed_profile_binding_accepts_profile_gain(self) -> None:
        guard = binding_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []
        train_calls: list[float] = []

        def train_branch_diversity(
            *_args: object,
            **_kwargs: object,
        ) -> float:
            train_calls.append(float(_args[2]))
            return 0.8

        def profile_delta(
            _snapshot: dict[str, object],
            _baseline: dict[str, object],
            profiles: list[str] | tuple[str, ...],
        ) -> dict[str, int]:
            if list(profiles) == ["qa:preserved"]:
                return {"regressed_profile_count": 0, "improved_profile_count": 0}
            return {"regressed_profile_count": 0, "improved_profile_count": 1}

        result = try_baseline_floor_collapsed_profile_binding(
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
            owner_paraphrase_binding_active=True,
            owner_paraphrase_target_profiles=("qa:owner",),
            owner_paraphrase_preserved_profiles=("qa:preserved",),
            owner_paraphrase_binding_preservation_delta=None,
            memory_consolidation_active=True,
            memory_consolidation_target_profiles=("qa:owner",),
            learning_rate_scales=(0.25,),
            select_binding_targets=lambda *_args, **_kwargs: ["qa:owner"],
            train_branch_diversity=train_branch_diversity,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (2.0,),
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_diversity_delta=profile_delta,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.loss_total, 0.8)
        self.assertEqual(result.loss_count, 1)
        self.assertEqual(result.learning_rate_scale, 0.25)
        self.assertEqual(result.target_profiles, ["qa:owner"])
        self.assertEqual(result.profile_score, (2.0,))
        self.assertEqual(result.diversity_outcome, "improved")
        self.assertEqual(len(restore_calls), 1)
        self.assertEqual(train_calls, [0.025])
        self.assertEqual(
            guard["profile_scale_collapsed_profile_binding_frontier_acceptances"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_preservation_checks"],
            1,
        )
        self.assertEqual(
            recorder.metadata[0][
                "baseline_floor_profile_scale_collapsed_profile_binding_probe"
            ],
            True,
        )
        self.assertEqual(
            recorder.metadata[0]["collapsed_profile_binding_target_profiles"],
            ["qa:owner"],
        )

    def test_collapsed_profile_binding_restores_after_preservation_failure(self) -> None:
        guard = binding_guard()
        recorder = FakeSnapshotRecorder()
        restore_calls: list[tuple[dict[str, object], dict[str, object]]] = []

        def profile_delta(
            _snapshot: dict[str, object],
            _baseline: dict[str, object],
            profiles: list[str] | tuple[str, ...],
        ) -> dict[str, int]:
            if list(profiles) == ["qa:preserved"]:
                return {"regressed_profile_count": 1, "improved_profile_count": 0}
            return {"regressed_profile_count": 0, "improved_profile_count": 1}

        result = try_baseline_floor_collapsed_profile_binding(
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
            owner_paraphrase_binding_active=True,
            owner_paraphrase_target_profiles=("qa:owner",),
            owner_paraphrase_preserved_profiles=("qa:preserved",),
            owner_paraphrase_binding_preservation_delta=None,
            memory_consolidation_active=False,
            memory_consolidation_target_profiles=(),
            learning_rate_scales=(0.25,),
            select_binding_targets=lambda *_args, **_kwargs: ["qa:owner"],
            train_branch_diversity=lambda *_args, **_kwargs: 0.8,
            preserves_target_coverage=lambda _snapshot, _baseline: True,
            snapshot_score=lambda _snapshot: (2.0,),
            target_coverage_delta=lambda _snapshot, _baseline: {
                "regressed_profile_count": 0,
                "improved_profile_count": 0,
            },
            profile_diversity_delta=profile_delta,
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.attempted)
        self.assertEqual(result.outcome, "preserved_profile_regressed")
        self.assertEqual(
            result.rejection_reason,
            "owner_paraphrase_preservation_regression",
        )
        self.assertEqual(len(restore_calls), 2)
        self.assertEqual(
            guard["profile_scale_collapsed_profile_binding_frontier_rejections"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_collapsed_profile_binding_frontier_rejection_reasons"
            ],
            {"owner_paraphrase_preservation_regression": 1},
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_preservation_failures"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_collapsed_profile_binding_frontier_fallback_acceptances"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
