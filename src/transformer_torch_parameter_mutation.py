"""Trainable-parameter mutation evidence for PyTorch parity probes."""

from __future__ import annotations

import math
from typing import Any


TORCH_PARAMETER_MUTATION_SCHEMA_VERSION = 1
TORCH_PARAMETER_MUTATION_OBSERVED_STATUS = "parameter_mutation_observed"
TORCH_PARAMETER_MUTATION_NOT_OBSERVED_STATUS = "parameter_mutation_not_observed"


def snapshot_torch_parameters(state: dict[str, Any]) -> dict[str, Any]:
    """Capture JSON-safe trainable-parameter signatures from tensor state."""

    parameters = [
        _parameter_snapshot(parameter)
        for parameter in state["parameters"]
    ]
    values = [
        value
        for parameter in parameters
        for value in parameter["_values"]
    ]
    return {
        "schema_version": TORCH_PARAMETER_MUTATION_SCHEMA_VERSION,
        "parameter_count": state["parameter_count"],
        "tensor_count": state["tensor_count"],
        "signature": _signature(values),
        "parameters": [
            {key: value for key, value in parameter.items() if key != "_values"}
            for parameter in parameters
        ],
        "_values_by_name": {
            parameter["name"]: parameter["_values"]
            for parameter in parameters
        },
    }


def build_torch_parameter_mutation_report(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compare two trainable-parameter snapshots."""

    before_parameters = {
        parameter["name"]: parameter
        for parameter in before["parameters"]
    }
    changed = [
        _parameter_change(
            before_parameters[parameter["name"]],
            parameter,
            before,
            after,
        )
        for parameter in after["parameters"]
    ]
    changed_scalar_count = sum(
        parameter["changed_scalar_count"] for parameter in changed
    )
    return {
        "schema_version": TORCH_PARAMETER_MUTATION_SCHEMA_VERSION,
        "status": (
            TORCH_PARAMETER_MUTATION_OBSERVED_STATUS
            if changed_scalar_count
            else TORCH_PARAMETER_MUTATION_NOT_OBSERVED_STATUS
        ),
        "before_signature": before["signature"],
        "after_signature": after["signature"],
        "changed_scalar_count": changed_scalar_count,
        "changed_tensor_count": sum(
            1 for parameter in changed if parameter["changed_scalar_count"]
        ),
        "max_abs_delta": max(
            (parameter["max_abs_delta"] for parameter in changed),
            default=0.0,
        ),
        "parameters": changed,
    }


def _parameter_snapshot(parameter: dict[str, Any]) -> dict[str, Any]:
    values = _tensor_values(parameter["tensor"])
    return {
        "name": parameter["name"],
        "count": len(values),
        "index_start": parameter["index_start"],
        "index_end": parameter["index_end"],
        "signature": _signature(values),
        "_values": values,
    }


def _parameter_change(
    before_parameter: dict[str, Any],
    after_parameter: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    name = after_parameter["name"]
    before_values = before["_values_by_name"][name]
    after_values = after["_values_by_name"][name]
    deltas = [
        abs(right - left)
        for left, right in zip(before_values, after_values)
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
    ]
    return {
        "name": name,
        "count": after_parameter["count"],
        "changed_scalar_count": len(deltas),
        "max_abs_delta": max(deltas, default=0.0),
        "before_signature": before_parameter["signature"],
        "after_signature": after_parameter["signature"],
    }


def _tensor_values(tensor: Any) -> list[float]:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    if hasattr(tensor, "tolist"):
        tensor = tensor.tolist()
    return list(_numbers(tensor))


def _signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


def _numbers(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, int | float):
        yield float(value)
