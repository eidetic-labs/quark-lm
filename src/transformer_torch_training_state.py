"""Trainable PyTorch state construction from scalar training fixtures."""

from __future__ import annotations

import copy
from typing import Any

from transformer_training_parameter_manifest import (
    validate_training_parameter_manifest,
)


TORCH_TRAINING_STATE_SCHEMA_VERSION = 1


def build_torch_training_state(
    *,
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Create trainable tensors that follow the scalar parameter manifest."""

    manifest = fixture["parameter_manifest"]
    validate_training_parameter_manifest(
        manifest,
        optimizer_state=fixture["training_case"]["optimizer_state"],
    )
    parameters = [
        _build_parameter(
            entry=entry,
            weights=fixture["initial_weights"],
            torch=torch,
            runtime=runtime,
        )
        for entry in manifest["entries"]
    ]
    return {
        "schema_version": TORCH_TRAINING_STATE_SCHEMA_VERSION,
        "parameter_order": manifest["parameter_order"],
        "parameter_count": manifest["parameter_count"],
        "tensor_count": len(parameters),
        "parameters": parameters,
    }


def summarize_torch_training_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe summary of a trainable tensor state."""

    return {
        "schema_version": state["schema_version"],
        "parameter_order": state["parameter_order"],
        "parameter_count": state["parameter_count"],
        "tensor_count": state["tensor_count"],
        "parameters": [
            {
                "name": parameter["name"],
                "shape": list(parameter["shape"]),
                "count": parameter["count"],
                "index_start": parameter["index_start"],
                "index_end": parameter["index_end"],
                "requires_grad": bool(
                    getattr(parameter["tensor"], "requires_grad", False)
                ),
            }
            for parameter in state["parameters"]
        ],
    }


def torch_training_weights_from_state(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Overlay trainable tensors onto the fixture's initial weight tree.

    The trainable tensors are persistent leaf objects updated in place by the
    optimizer, so the overlaid tree is built once and cached on the state: every
    forward reads the current tensor values through the same tree. This avoids a
    full deepcopy of the weight tree on every training step, which dominated
    runtime for large configs (e.g. context-size-48 prompt-position projection).
    """

    cached = state.get("weight_tree")
    if cached is not None:
        return cached
    weights = copy.deepcopy(fixture["initial_weights"])
    for parameter in state["parameters"]:
        _assign_weight(weights, parameter["name"], parameter["tensor"])
    state["weight_tree"] = weights
    return weights


def validate_torch_training_state_summary(
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Validate a JSON-safe state summary against a scalar parameter manifest."""

    if summary.get("schema_version") != TORCH_TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported torch training state schema_version")
    for key in ("parameter_order", "parameter_count", "tensor_count"):
        if summary.get(key) != manifest.get(key):
            raise ValueError(f"torch training state {key} does not match manifest")
    parameters = summary.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("torch training state parameters must be a list")
    if len(parameters) != len(manifest["entries"]):
        raise ValueError("torch training state parameter length mismatch")
    for actual, expected in zip(parameters, manifest["entries"]):
        _validate_summary_parameter(actual, expected)


def _build_parameter(
    *,
    entry: dict[str, Any],
    weights: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    value = _resolve_weight(weights, entry["name"])
    shape = _shape(value)
    if shape != entry["shape"]:
        raise ValueError(f"weight shape for {entry['name']} does not match manifest")
    return {
        "name": entry["name"],
        "shape": list(entry["shape"]),
        "count": entry["count"],
        "index_start": entry["index_start"],
        "index_end": entry["index_end"],
        "tensor": torch.tensor(
            value,
            dtype=getattr(torch, runtime["dtype"]),
            device=runtime["device"],
            requires_grad=True,
        ),
    }


def _resolve_weight(weights: dict[str, Any], name: str) -> Any:
    current: Any = weights
    for segment in name.split("."):
        if isinstance(current, list):
            current = current[int(segment)]
        else:
            current = current[segment]
    return current


def _assign_weight(weights: dict[str, Any], name: str, value: Any) -> None:
    segments = name.split(".")
    current: Any = weights
    for segment in segments[:-1]:
        current = (
            current[int(segment)]
            if isinstance(current, list)
            else current[segment]
        )
    final_segment = segments[-1]
    if isinstance(current, list):
        current[int(final_segment)] = value
    else:
        current[final_segment] = value


def _shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    return [len(value), *_shape(value[0])]


def _validate_summary_parameter(
    actual: Any,
    expected: dict[str, Any],
) -> None:
    if not isinstance(actual, dict):
        raise ValueError("torch training state parameter must be a dict")
    for key in ("name", "shape", "count", "index_start", "index_end"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"torch training state parameter {key} mismatch")
    if actual.get("requires_grad") is not True:
        raise ValueError("torch training state parameter must require grad")
