"""JSON-safe PyTorch gradient snapshots for parity probes."""

from __future__ import annotations

from typing import Any


TORCH_GRADIENT_SNAPSHOT_SCHEMA_VERSION = 1


def snapshot_torch_gradients(state: dict[str, Any]) -> dict[str, Any]:
    """Flatten current tensor gradients into a deterministic evidence artifact."""

    parameters = [_parameter_gradient(parameter) for parameter in state["parameters"]]
    values = [
        value
        for parameter in parameters
        for value in parameter["values"]
    ]
    return {
        "schema_version": TORCH_GRADIENT_SNAPSHOT_SCHEMA_VERSION,
        "parameter_count": len(parameters),
        "gradient_tensor_count": sum(
            1 for parameter in parameters if parameter["has_gradient"]
        ),
        "scalar_count": len(values),
        "signature": _signature(values),
        "parameters": parameters,
    }


def _parameter_gradient(parameter: dict[str, Any]) -> dict[str, Any]:
    values = _gradient_values(parameter["tensor"])
    return {
        "name": parameter["name"],
        "has_gradient": bool(values),
        "count": len(values),
        "values": values,
        "signature": _signature(values),
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


def _signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }
