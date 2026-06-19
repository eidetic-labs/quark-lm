"""Scalar optimizer gradient-buffer evidence for training parity fixtures."""

from __future__ import annotations

from typing import Any


OPTIMIZER_GRADIENT_EVIDENCE_SCHEMA_VERSION = 1


def build_optimizer_gradient_evidence(
    *,
    raw_gradients: list[float],
    clipped_gradients: list[float],
    buffer_before: list[float],
    buffer_after_add: list[float],
    accumulated_gradients: list[float] | None,
    update_applied: bool,
    update_count_before: int,
    update_count_after: int,
    pending_accumulation_before: int,
    pending_accumulation_after: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Build JSON-safe evidence for one scalar optimizer application."""

    return {
        "schema_version": OPTIMIZER_GRADIENT_EVIDENCE_SCHEMA_VERSION,
        "raw_gradient": _vector_evidence(raw_gradients),
        "clipped_gradient": _vector_evidence(clipped_gradients),
        "buffer_before": _vector_evidence(buffer_before),
        "buffer_after_add": _vector_evidence(buffer_after_add),
        "accumulated_gradient": _optional_vector_evidence(
            accumulated_gradients
        ),
        "update_applied": update_applied,
        "learning_rate": learning_rate,
        "update_count_before": update_count_before,
        "update_count_after": update_count_after,
        "pending_accumulation_before": pending_accumulation_before,
        "pending_accumulation_after": pending_accumulation_after,
    }


def _optional_vector_evidence(values: list[float] | None) -> dict[str, Any]:
    if values is None:
        return {
            "available": False,
            "values": [],
            "signature": _signature([]),
        }
    return {
        "available": True,
        "values": list(values),
        "signature": _signature(values),
    }


def _vector_evidence(values: list[float]) -> dict[str, Any]:
    return {
        "values": list(values),
        "signature": _signature(values),
    }


def _signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }
