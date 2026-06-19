"""Gradient clipping evidence for optional PyTorch training parity."""

from __future__ import annotations

import math
from typing import Any


TORCH_GRADIENT_CLIP_SCHEMA_VERSION = 1
TORCH_GRADIENT_CLIP_APPLIED_STATUS = "gradient_clip_applied"


def apply_torch_gradient_value_clip(
    *,
    torch: Any,
    state: dict[str, Any],
    clip_value: float,
) -> dict[str, Any]:
    """Apply PyTorch value clipping and report JSON-safe mutation evidence."""

    if clip_value <= 0.0:
        return {
            "schema_version": TORCH_GRADIENT_CLIP_SCHEMA_VERSION,
            "status": "not_required",
            "applied": False,
            "value": clip_value,
            "reason": "gradient clipping is disabled",
        }
    clipper = _clip_grad_value(torch)
    if not callable(clipper):
        return {
            "schema_version": TORCH_GRADIENT_CLIP_SCHEMA_VERSION,
            "status": "clipper_unavailable",
            "applied": False,
            "value": clip_value,
            "reason": "torch.nn.utils.clip_grad_value_ is not available",
        }

    before = _gradient_values_by_parameter(state)
    clipper([parameter["tensor"] for parameter in state["parameters"]], clip_value)
    after = _gradient_values_by_parameter(state)
    parameters = [
        _parameter_clip_summary(name, before[name], after[name])
        for name in before
    ]
    return {
        "schema_version": TORCH_GRADIENT_CLIP_SCHEMA_VERSION,
        "status": TORCH_GRADIENT_CLIP_APPLIED_STATUS,
        "applied": True,
        "value": clip_value,
        "gradient_tensor_count": sum(
            1 for parameter in parameters if parameter["has_gradient"]
        ),
        "max_abs_before": max(
            (parameter["max_abs_before"] for parameter in parameters),
            default=0.0,
        ),
        "max_abs_after": max(
            (parameter["max_abs_after"] for parameter in parameters),
            default=0.0,
        ),
        "changed_scalar_count": sum(
            parameter["changed_scalar_count"] for parameter in parameters
        ),
        "parameters": parameters,
    }


def _clip_grad_value(torch: Any) -> Any:
    nn = getattr(torch, "nn", None)
    utils = getattr(nn, "utils", None) if nn is not None else None
    return getattr(utils, "clip_grad_value_", None)


def _gradient_values_by_parameter(state: dict[str, Any]) -> dict[str, list[float]]:
    return {
        parameter["name"]: _gradient_values(parameter["tensor"])
        for parameter in state["parameters"]
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


def _parameter_clip_summary(
    name: str,
    before: list[float],
    after: list[float],
) -> dict[str, Any]:
    return {
        "name": name,
        "has_gradient": bool(before),
        "max_abs_before": _max_abs(before),
        "max_abs_after": _max_abs(after),
        "changed_scalar_count": _changed_scalar_count(before, after),
    }


def _max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def _changed_scalar_count(before: list[float], after: list[float]) -> int:
    return sum(
        1
        for left, right in zip(before, after)
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
    )


def _numbers(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, int | float):
        yield float(value)
