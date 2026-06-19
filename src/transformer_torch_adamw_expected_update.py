"""Expected AdamW update evidence for PyTorch training parity probes."""

from __future__ import annotations

import math
from typing import Any


TORCH_ADAMW_EXPECTED_UPDATE_SCHEMA_VERSION = 1
TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS = "adamw_expected_update_built"


def build_torch_adamw_expected_update(
    *,
    state: dict[str, Any],
    parameters_before: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Compute the scalar AdamW signature expected from current gradients."""

    if contract["optimizer"] != "adamw":
        return _not_built(f"unsupported optimizer: {contract['optimizer']}")
    applied = [
        record
        for record in contract["expected_step_records"]
        if record["update_applied"]
    ]
    if len(applied) != 1:
        return _not_built("expected AdamW replay requires exactly one update")
    record = applied[0]
    if record["update_count_after"] != 1:
        return _not_built("expected AdamW replay requires zero prior moments")

    update = _updated_values(
        state=state,
        parameters_before=parameters_before,
        adamw=contract["adamw"],
        learning_rate=record["effective_learning_rate"],
        update_count=record["update_count_after"],
    )
    if update["error"] is not None:
        return _not_built(update["error"])
    return {
        "schema_version": TORCH_ADAMW_EXPECTED_UPDATE_SCHEMA_VERSION,
        "status": TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS,
        "optimizer": "adamw",
        "parameter_count": contract["parameter_count"],
        "effective_learning_rate": record["effective_learning_rate"],
        "update_count": record["update_count_after"],
        "assumptions": {
            "prior_moments": "zero",
            "gradient_source": contract["gradient_source"],
            "gradients_already_clipped": True,
        },
        "expected_signature": _signature(update["values"]),
        "gradient_signature": _signature(update["gradients"]),
    }


def _updated_values(
    *,
    state: dict[str, Any],
    parameters_before: dict[str, Any],
    adamw: dict[str, float],
    learning_rate: float,
    update_count: int,
) -> dict[str, Any]:
    values = []
    gradients = []
    for parameter in state["parameters"]:
        name = parameter["name"]
        before = parameters_before["_values_by_name"][name]
        grads = _gradient_values(parameter["tensor"])
        if len(before) != len(grads):
            return {
                "error": f"gradient count mismatch for {name}",
                "values": [],
                "gradients": [],
            }
        values.extend(
            _adamw_value(
                value=value,
                grad=grad,
                adamw=adamw,
                learning_rate=learning_rate,
                update_count=update_count,
            )
            for value, grad in zip(before, grads)
        )
        gradients.extend(grads)
    return {
        "error": None,
        "values": values,
        "gradients": gradients,
    }


def _adamw_value(
    *,
    value: float,
    grad: float,
    adamw: dict[str, float],
    learning_rate: float,
    update_count: int,
) -> float:
    beta1 = adamw["beta1"]
    beta2 = adamw["beta2"]
    first_moment = (1.0 - beta1) * grad
    second_moment = (1.0 - beta2) * grad * grad
    first_unbiased = first_moment / (1.0 - beta1**update_count)
    second_unbiased = second_moment / (1.0 - beta2**update_count)
    updated = value
    if adamw["weight_decay"] > 0.0:
        updated -= learning_rate * adamw["weight_decay"] * updated
    return updated - (
        learning_rate
        * first_unbiased
        / (math.sqrt(second_unbiased) + adamw["epsilon"])
    )


def _gradient_values(tensor: Any) -> list[float]:
    grad = getattr(tensor, "grad", None)
    if grad is None:
        return []
    if hasattr(grad, "detach"):
        grad = grad.detach().cpu()
    if hasattr(grad, "tolist"):
        grad = grad.tolist()
    return list(_numbers(grad))


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


def _not_built(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_ADAMW_EXPECTED_UPDATE_SCHEMA_VERSION,
        "status": "not_built",
        "reason": reason,
    }
