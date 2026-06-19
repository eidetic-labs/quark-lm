"""PyTorch microstep replay plans for accumulated-gradient parity."""

from __future__ import annotations

from typing import Any


TORCH_ACCUMULATION_REPLAY_PLAN_SCHEMA_VERSION = 1
TORCH_ACCUMULATION_REPLAY_PENDING_STATUS = "accumulation_replay_pending"


def build_torch_accumulation_replay_plan(
    *,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    """Describe the PyTorch backward replay needed to match scalar training."""

    case = fixture["training_case"]
    contract = fixture["optimizer_step_contract"]
    accumulation = contract["gradient_accumulation"]
    equivalence = accumulation["pytorch_equivalence"]
    microsteps = [
        _microstep(
            case=case,
            record=record,
            accumulation=accumulation,
            equivalence=equivalence,
        )
        for record in contract["expected_step_records"]
    ]
    return {
        "schema_version": TORCH_ACCUMULATION_REPLAY_PLAN_SCHEMA_VERSION,
        "status": TORCH_ACCUMULATION_REPLAY_PENDING_STATUS,
        "reason": "microstep replay is planned but not executed",
        "case_id": case["case_id"],
        "microstep_count": len(microsteps),
        "accumulation_steps": contract["gradient_accumulation_steps"],
        "gradient_source": accumulation["gradient_source"],
        "reduction": accumulation["reduction"],
        "requires_microstep_clipping": accumulation[
            "requires_microstep_clipping"
        ],
        "requires_clipped_gradient_buffer": equivalence[
            "requires_clipped_gradient_buffer"
        ],
        "native_loss_scaling_sufficient": equivalence[
            "native_loss_scaling_sufficient"
        ],
        "accumulated_gradient_parity_proven": False,
        "microsteps": microsteps,
        "execution_status": {
            "replayed_backward_passes": False,
            "buffered_gradients": False,
            "applied_optimizer_updates": False,
        },
    }


def _microstep(
    *,
    case: dict[str, Any],
    record: dict[str, Any],
    accumulation: dict[str, Any],
    equivalence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": record["step"],
        "context": list(case["context"]),
        "target": case["target"],
        "loss_scale": _loss_scale(
            equivalence=equivalence,
        ),
        "backward_required": True,
        "clip_after_backward": accumulation["requires_microstep_clipping"],
        "buffer_action": _buffer_action(equivalence),
        "reduce_buffer_before_update": record["update_applied"],
        "optimizer_step_after_microstep": record["update_applied"],
        "optimizer_zero_grad_after_microstep": record["update_applied"],
        "effective_learning_rate": record["effective_learning_rate"],
        "pending_accumulation_before": record["pending_accumulation_before"],
        "pending_accumulation_after": record["pending_accumulation_after"],
        "update_count_before": record["update_count_before"],
        "update_count_after": record["update_count_after"],
    }


def _loss_scale(*, equivalence: dict[str, Any]) -> float:
    if equivalence["requires_clipped_gradient_buffer"]:
        return 1.0
    return float(equivalence["loss_scale_if_no_microstep_clipping"])


def _buffer_action(equivalence: dict[str, Any]) -> str:
    if equivalence["requires_clipped_gradient_buffer"]:
        return "clip_then_buffer_gradient"
    return "accumulate_scaled_gradient"
