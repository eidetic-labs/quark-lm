"""Backward-execution probes for optional PyTorch training parity."""

from __future__ import annotations

import math
from typing import Any

from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_tensor_ops import torch_to_float
from transformer_torch_training_state import torch_training_weights_from_state


TORCH_TRAINING_BACKWARD_PROBE_SCHEMA_VERSION = 1


def build_torch_training_backward_probe(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tensor loss backward pass and summarize gradient coverage."""

    case = fixture["training_case"]
    _clear_gradients(state)
    loss = _loss_tensor(fixture=fixture, state=state, torch=torch, runtime=runtime)
    if not callable(getattr(loss, "backward", None)):
        return _probe_result(
            fixture=fixture,
            status="backward_unavailable",
            loss_value=torch_to_float(loss),
            gradient_summary=_gradient_summary(state),
        )
    loss.backward()
    return _probe_result(
        fixture=fixture,
        status=_status(state),
        loss_value=torch_to_float(loss),
        gradient_summary=_gradient_summary(state),
    )


def _loss_tensor(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    case = fixture["training_case"]
    weights = torch_training_weights_from_state(fixture=fixture, state=state)
    logits = torch_minimal_logits(
        case["context"],
        {
            "weights": weights,
            "model_config": fixture["model_config"],
        },
        torch,
        runtime,
    )
    probabilities = torch.softmax(logits, dim=0)
    return -torch.log(probabilities[case["target"]])


def _probe_result(
    *,
    fixture: dict[str, Any],
    status: str,
    loss_value: float,
    gradient_summary: dict[str, Any],
) -> dict[str, Any]:
    case = fixture["training_case"]
    return {
        "schema_version": TORCH_TRAINING_BACKWARD_PROBE_SCHEMA_VERSION,
        "status": status,
        "case_id": case["case_id"],
        "loss": loss_value,
        "loss_abs_diff": abs(case["initial_loss"] - loss_value),
        "loss_matches_initial": math.isclose(
            case["initial_loss"],
            loss_value,
            abs_tol=fixture["tolerance"]["absolute"],
            rel_tol=fixture["tolerance"]["relative"],
        ),
        "gradient_summary": gradient_summary,
    }


def _clear_gradients(state: dict[str, Any]) -> None:
    for parameter in state["parameters"]:
        tensor = parameter["tensor"]
        if hasattr(tensor, "grad"):
            tensor.grad = None


def _status(state: dict[str, Any]) -> str:
    summary = _gradient_summary(state)
    if summary["missing_gradient_tensor_count"] == 0:
        return "gradients_available"
    return "gradients_missing"


def _gradient_summary(state: dict[str, Any]) -> dict[str, Any]:
    parameters = state["parameters"]
    missing = [
        parameter["name"]
        for parameter in parameters
        if getattr(parameter["tensor"], "grad", None) is None
    ]
    return {
        "tensor_count": len(parameters),
        "gradient_tensor_count": len(parameters) - len(missing),
        "missing_gradient_tensor_count": len(missing),
        "missing_gradient_parameters": missing,
    }
