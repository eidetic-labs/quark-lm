from __future__ import annotations

import unittest
from types import SimpleNamespace

from transformer_routing_repair_update_search import (
    ROUTING_REPAIR_LEARNING_RATE_SCALES,
    RoutingRepairUpdateSearchContext,
    apply_routing_repair_update_search,
)


class TransformerRoutingRepairUpdateSearchTests(unittest.TestCase):
    def test_accepts_first_coverage_preserving_retry_scale(self) -> None:
        recorder = _Recorder([_snapshot(0.25), _snapshot(0.75)])
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = _guard()

        result = apply_routing_repair_update_search(
            _context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(train_rates, [0.04, 0.02])
        self.assertEqual(len(restores), 2)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], 2)
        self.assertEqual(guard["rejected_attempts"], 1)
        self.assertEqual(guard["accepted_steps"], 1)
        self.assertEqual(guard["rejected_steps"], 0)
        self.assertEqual(guard["routing_repair_accepted_learning_rate_scale"], 0.5)

    def test_rejects_preserved_coverage_without_response(self) -> None:
        recorder = _Recorder(
            [_snapshot(0.5)] * len(ROUTING_REPAIR_LEARNING_RATE_SCALES)
        )
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = _guard()

        result = apply_routing_repair_update_search(
            _context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(guard["accepted_steps"], 0)
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertEqual(
            guard["routing_repair_no_response_rejections"],
            len(ROUTING_REPAIR_LEARNING_RATE_SCALES),
        )
        self.assertIsNone(guard["routing_repair_accepted_learning_rate_scale"])

    def test_accepts_preserved_coverage_with_rank_response(self) -> None:
        recorder = _Recorder([_snapshot(0.5, target_rank=8.0, top3_rate=0.25)])
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = _guard()

        result = apply_routing_repair_update_search(
            _context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(guard["accepted_steps"], 1)
        self.assertEqual(guard["rejected_steps"], 0)
        self.assertEqual(
            guard["routing_repair_branch_response_acceptances"],
            1,
        )
        self.assertEqual(guard["routing_repair_accepted_learning_rate_scale"], 1.0)

    def test_rejects_all_scales_and_restores_baseline(self) -> None:
        recorder = _Recorder([_snapshot(0.25)] * len(ROUTING_REPAIR_LEARNING_RATE_SCALES))
        restores: list[tuple[dict, dict]] = []
        train_rates: list[float] = []
        guard = _guard()

        result = apply_routing_repair_update_search(
            _context(
                recorder=recorder,
                guard=guard,
                restores=restores,
                train_rates=train_rates,
            )
        )

        self.assertTrue(result.update_guard_applied)
        self.assertEqual(len(train_rates), len(ROUTING_REPAIR_LEARNING_RATE_SCALES))
        self.assertEqual(len(restores), len(ROUTING_REPAIR_LEARNING_RATE_SCALES) + 1)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], len(ROUTING_REPAIR_LEARNING_RATE_SCALES))
        self.assertEqual(guard["rejected_attempts"], len(ROUTING_REPAIR_LEARNING_RATE_SCALES))
        self.assertEqual(guard["accepted_steps"], 0)
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertIsNone(guard["routing_repair_accepted_learning_rate_scale"])


def _context(
    *,
    recorder: "_Recorder",
    guard: dict,
    restores: list[tuple[dict, dict]],
    train_rates: list[float],
) -> RoutingRepairUpdateSearchContext:
    return RoutingRepairUpdateSearchContext(
        args=SimpleNamespace(direct_answer_learning_rate=0.04),
        direct_step=1,
        example="example",
        lesson="lesson",
        branch_examples=["example"],
        rng=_Rng(),
        terminator="\n",
        direct_baseline=_snapshot(0.5),
        direct_snapshot_recorder=recorder,
        direct_answer_update_guard=guard,
        model=lambda: object(),
        tokenizer=lambda: object(),
        params=lambda: [],
        restore_state=lambda model, optimizer: restores.append((model, optimizer)),
        train_mode_step=lambda **kwargs: _train_result(kwargs, train_rates),
        train_adaptive_baseline_floor_update=lambda *args: 0.0,
        train_baseline_anchored_prompt=lambda *args: 0.0,
        pre_update_model_payload={"model": True},
        pre_update_optimizer_payload={"optimizer": True},
        pre_update_rng_state="rng-state",
    )


def _train_result(kwargs: dict, train_rates: list[float]) -> SimpleNamespace:
    train_rates.append(float(kwargs["args"].direct_answer_learning_rate))
    return SimpleNamespace(loss=train_rates[-1], update_guard_applied=False)


def _snapshot(
    coverage: float,
    *,
    target_rank: float = 20.0,
    top3_rate: float = 0.0,
) -> dict:
    return {
        "branch_diversity_target": {
            "passed": False,
            "passed_profiles": 0,
            "failed_profiles": 1,
            "min_target_token_coverage": coverage,
        },
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 2,
                    "target_token_coverage": coverage,
                    "predicted_unique": 1,
                    "dominant_predicted_rate": 1.0,
                },
                "target_rank": {
                    "avg": target_rank,
                    "top3_rate": top3_rate,
                    "top5_rate": top3_rate,
                }
            }
        }
    }


def _guard() -> dict:
    return {
        "checked_steps": 0,
        "attempted_updates": 0,
        "accepted_steps": 0,
        "accepted_attempts": 0,
        "repaired_steps": 0,
        "repaired_attempts": 0,
        "stabilized_steps": 0,
        "stabilized_attempts": 0,
        "accepted_learning_rate_scale_counts": {},
        "accepted_update_shape_counts": {},
        "rejected_steps": 0,
        "rejected_attempts": 0,
        "rejected_learning_rate_scale_counts": {},
        "rejected_update_shape_counts": {},
        "rejected_violation_profile_counts": {},
        "worst_rejected_coverage_deficit": 0.0,
        "worst_rejected_coverage_violation": None,
        "rejected_floor_diagnostic_sample": [],
        "rejected_step_sample": [],
    }


class _Recorder:
    def __init__(self, snapshots: list[dict]) -> None:
        self.snapshots = snapshots
        self.index = 0

    def record(
        self,
        step: int,
        train_loss: float | None,
        extra: dict | None = None,
    ) -> dict:
        snapshot = dict(self.snapshots[self.index])
        self.index += 1
        return snapshot


class _Rng:
    def __init__(self) -> None:
        self.restored_states: list[object] = []

    def setstate(self, state: object) -> None:
        self.restored_states.append(state)


if __name__ == "__main__":
    unittest.main()
