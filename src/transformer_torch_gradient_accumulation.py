"""Accumulated-gradient scope evidence for optional PyTorch parity probes."""

from __future__ import annotations

from typing import Any

from transformer_torch_accumulation_readiness import (
    build_torch_accumulation_readiness,
)


TORCH_GRADIENT_ACCUMULATION_SCHEMA_VERSION = 1
TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS = "gradient_accumulation_recorded"


def build_torch_gradient_accumulation_report(
    *,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Summarize accumulation cadence without claiming gradient replay parity."""

    records = contract["expected_step_records"]
    accumulation = contract["gradient_accumulation"]
    expected_update_count = sum(1 for record in records if record["update_applied"])
    pending_step_count = sum(1 for record in records if not record["update_applied"])
    requires_replay = _requires_replayed_backward_passes(contract)
    return {
        "schema_version": TORCH_GRADIENT_ACCUMULATION_SCHEMA_VERSION,
        "status": TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS,
        "gradient_source": contract["gradient_source"],
        "gradient_scope": "current_tensor_grad_sample",
        "accumulation_steps": contract["gradient_accumulation_steps"],
        "reduction": accumulation["reduction"],
        "microstep_gradient_source": accumulation["gradient_source"],
        "requires_microstep_clipping": accumulation[
            "requires_microstep_clipping"
        ],
        "pytorch_equivalence": accumulation["pytorch_equivalence"],
        "pytorch_accumulation_readiness": build_torch_accumulation_readiness(
            contract=contract,
        ),
        "expected_step_count": len(records),
        "expected_update_count": expected_update_count,
        "pending_step_count": pending_step_count,
        "applied_update_count": expected_update_count,
        "final_pending_accumulation": contract["expected_final_optimizer_state"][
            "pending_accumulation"
        ],
        "uses_single_gradient_sample": True,
        "requires_replayed_backward_passes": requires_replay,
        "accumulated_gradient_parity_proven": False,
        "accumulated_gradient_parity_status": _parity_status(requires_replay),
        "reason": _reason(requires_replay),
        "current_gradient_signature": _gradient_signature(state),
        "step_records": [_step_record(record) for record in records],
    }


def _requires_replayed_backward_passes(contract: dict[str, Any]) -> bool:
    return (
        contract["gradient_accumulation_steps"] > 1
        or len(contract["expected_step_records"]) > 1
    )


def _parity_status(requires_replay: bool) -> str:
    if requires_replay:
        return "not_proven"
    return "not_required"


def _reason(requires_replay: bool) -> str:
    if requires_replay:
        return (
            "current probe records one tensor.grad sample; PyTorch must replay "
            "backward passes per microstep before accumulated-gradient parity "
            "can be claimed"
        )
    return "no multi-step accumulation replay is required for this contract"


def _gradient_signature(state: dict[str, Any]) -> dict[str, float | int]:
    values = []
    gradient_parameter_count = 0
    missing_gradient_parameter_count = 0
    for parameter in state["parameters"]:
        parameter_values = _gradient_values(parameter["tensor"])
        if parameter_values:
            gradient_parameter_count += 1
            values.extend(parameter_values)
        else:
            missing_gradient_parameter_count += 1
    return {
        "parameter_count": len(state["parameters"]),
        "gradient_parameter_count": gradient_parameter_count,
        "missing_gradient_parameter_count": missing_gradient_parameter_count,
        "scalar_count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


def _gradient_values(tensor: Any) -> list[float]:
    grad = getattr(tensor, "grad", None)
    if grad is None:
        return []
    if hasattr(grad, "detach"):
        grad = grad.detach().cpu()
    if hasattr(grad, "tolist"):
        grad = grad.tolist()
    return list(_numbers(grad))


def _numbers(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, int | float):
        yield float(value)


def _step_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": record["step"],
        "update_applied": record["update_applied"],
        "update_count_before": record["update_count_before"],
        "update_count_after": record["update_count_after"],
        "pending_accumulation_before": record["pending_accumulation_before"],
        "pending_accumulation_after": record["pending_accumulation_after"],
        "effective_learning_rate": record["effective_learning_rate"],
    }
