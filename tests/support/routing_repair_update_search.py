"""Fixtures for routing-repair update-search tests."""

from __future__ import annotations

from types import SimpleNamespace

from transformer_routing_repair_update_search import RoutingRepairUpdateSearchContext


def routing_repair_context(
    *,
    recorder: "RoutingRepairRecorder",
    guard: dict,
    restores: list[tuple[dict, dict]],
    train_rates: list[float],
    baseline: dict | None = None,
) -> RoutingRepairUpdateSearchContext:
    return RoutingRepairUpdateSearchContext(
        args=SimpleNamespace(direct_answer_learning_rate=0.04),
        direct_step=1,
        example="example",
        lesson="lesson",
        branch_examples=["example"],
        rng=_Rng(),
        terminator="\n",
        direct_baseline=baseline or routing_repair_snapshot(0.5),
        direct_snapshot_recorder=recorder,
        direct_answer_update_guard=guard,
        model=lambda: object(),
        tokenizer=lambda: object(),
        optimizer=lambda: _Optimizer(),
        params=lambda: [],
        restore_state=lambda model, optimizer: restores.append((model, optimizer)),
        train_mode_step=lambda **kwargs: _train_result(kwargs, train_rates),
        train_adaptive_baseline_floor_update=lambda *args: 0.0,
        train_baseline_anchored_prompt=lambda *args: 0.0,
        pre_update_model_payload={"model": True},
        pre_update_optimizer_payload={"optimizer": True},
        pre_update_rng_state="rng-state",
    )


def routing_repair_snapshot(
    coverage: float,
    *,
    target_rank: float = 20.0,
    top3_rate: float = 0.0,
    predicted_unique: int = 1,
    dominant_rate: float = 1.0,
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
                    "predicted_unique": predicted_unique,
                    "dominant_predicted_rate": dominant_rate,
                },
                "target_rank": {
                    "avg": target_rank,
                    "top3_rate": top3_rate,
                    "top5_rate": top3_rate,
                },
            }
        },
    }


def routing_repair_guard() -> dict:
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
        "rejected_stability_violation_counts": {},
        "rejected_stability_violation_profile_counts": {},
        "worst_rejected_coverage_deficit": 0.0,
        "worst_rejected_coverage_violation": None,
        "rejected_floor_diagnostic_sample": [],
        "rejected_stability_diagnostic_sample": [],
        "rejected_step_sample": [],
    }


class RoutingRepairRecorder:
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


def _train_result(kwargs: dict, train_rates: list[float]) -> SimpleNamespace:
    train_rates.append(float(kwargs["args"].direct_answer_learning_rate))
    return SimpleNamespace(loss=train_rates[-1], update_guard_applied=False)


class _Rng:
    def __init__(self) -> None:
        self.restored_states: list[object] = []

    def setstate(self, state: object) -> None:
        self.restored_states.append(state)


class _Optimizer:
    last_apply_evidence = {
        "raw_gradient": {"signature": {"abs_sum": 1.5}},
        "clipped_gradient": {"signature": {"abs_sum": 1.25}},
        "accumulated_gradient": {
            "available": True,
            "signature": {"abs_sum": 1.0},
        },
        "update_applied": True,
        "learning_rate": 0.04,
        "update_count_before": 0,
        "update_count_after": 1,
        "pending_accumulation_before": 0,
        "pending_accumulation_after": 0,
    }
