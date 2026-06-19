"""Optimizer-step readiness probes for optional PyTorch training parity."""

from __future__ import annotations

from typing import Any

from transformer_optimizer_step_contract import validate_optimizer_step_contract


TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION = 1
TORCH_OPTIMIZER_STEP_READY_STATUS = "ready_for_optimizer_execution"


def build_torch_optimizer_step_probe(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any] | None,
    backward_probe: dict[str, Any],
) -> dict[str, Any]:
    """Summarize whether tensor gradients satisfy the optimizer contract."""

    if state is None:
        return _not_run("pytorch training runtime is not ready")
    if backward_probe.get("status") != "gradients_available":
        return _not_run("pytorch gradients are not available")

    contract = fixture["optimizer_step_contract"]
    try:
        validate_optimizer_step_contract(
            contract,
            training_case=fixture["training_case"],
        )
    except ValueError as exc:
        return {
            "schema_version": TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION,
            "status": "contract_invalid",
            "reason": str(exc),
        }

    summary = summarize_torch_optimizer_gradients(
        state=state,
        contract=contract,
    )
    status, reason = _readiness_status(summary)
    return {
        "schema_version": TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "optimizer": contract["optimizer"],
        "gradient_source": contract["gradient_source"],
        "expected_update_count": contract["expected_final_optimizer_state"][
            "update_count"
        ],
        "expected_step_count": len(contract["expected_step_records"]),
        "gradient_summary": summary,
    }


def summarize_torch_optimizer_gradients(
    *,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-safe gradient coverage summary for optimizer readiness."""

    parameters = [
        _parameter_gradient_summary(parameter)
        for parameter in state["parameters"]
    ]
    missing = [
        parameter["name"]
        for parameter in parameters
        if not parameter["has_gradient"]
    ]
    mismatched = [
        parameter["name"]
        for parameter in parameters
        if parameter["has_gradient"] and not parameter["shape_matches"]
    ]
    return {
        "tensor_count": len(parameters),
        "gradient_tensor_count": len(parameters) - len(missing),
        "missing_gradient_tensor_count": len(missing),
        "missing_gradient_parameters": missing,
        "shape_mismatch_count": len(mismatched),
        "shape_mismatch_parameters": mismatched,
        "gradient_parameter_count": sum(
            parameter["count"]
            for parameter in parameters
            if parameter["has_gradient"]
        ),
        "expected_parameter_count": contract["parameter_count"],
        "parameter_order": state["parameter_order"],
        "parameter_order_matches": state["parameter_order"]
        == contract["parameter_order"],
        "parameter_index_coverage_matches": _index_coverage_matches(
            parameters,
            contract["parameter_count"],
        ),
        "parameters": parameters,
    }


def _parameter_gradient_summary(parameter: dict[str, Any]) -> dict[str, Any]:
    grad = getattr(parameter["tensor"], "grad", None)
    gradient_shape = None if grad is None else _shape(grad)
    expected_shape = list(parameter["shape"])
    return {
        "name": parameter["name"],
        "shape": expected_shape,
        "gradient_shape": gradient_shape,
        "count": parameter["count"],
        "index_start": parameter["index_start"],
        "index_end": parameter["index_end"],
        "has_gradient": grad is not None,
        "shape_matches": gradient_shape == expected_shape,
    }


def _readiness_status(summary: dict[str, Any]) -> tuple[str, str]:
    if summary["missing_gradient_tensor_count"] > 0:
        return "missing_gradients", "one or more optimizer tensors lack gradients"
    if summary["shape_mismatch_count"] > 0:
        return "gradient_shape_mismatch", "one or more gradients have wrong shape"
    if not summary["parameter_order_matches"]:
        return "parameter_order_mismatch", "state order does not match contract"
    if not summary["parameter_index_coverage_matches"]:
        return "parameter_index_mismatch", "state indexes do not cover contract"
    if summary["gradient_parameter_count"] != summary["expected_parameter_count"]:
        return "gradient_count_mismatch", "gradient parameter count differs"
    return (
        TORCH_OPTIMIZER_STEP_READY_STATUS,
        "optimizer gradients satisfy the scalar step contract",
    )


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION,
        "status": "not_run",
        "reason": reason,
    }


def _index_coverage_matches(
    parameters: list[dict[str, Any]],
    parameter_count: int,
) -> bool:
    expected_start = 0
    for parameter in parameters:
        if parameter["index_start"] != expected_start:
            return False
        expected_start = parameter["index_end"]
    return expected_start == parameter_count


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(dimension) for dimension in shape]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    return [len(value), *_shape(value[0])]
