"""Optimizer diagnostics for routing-repair retry probes."""

from __future__ import annotations

from typing import Any

MAX_ROUTING_REPAIR_OPTIMIZER_PROBE_SAMPLES = 8


def record_routing_repair_optimizer_probe(
    guard: dict[str, Any],
    optimizer: Any,
    learning_rate_scale: float,
    loss: float,
) -> None:
    evidence = getattr(optimizer, "last_apply_evidence", None)
    if not isinstance(evidence, dict):
        guard["routing_repair_optimizer_probe_missing_evidence"] = int(
            guard.get("routing_repair_optimizer_probe_missing_evidence", 0)
        ) + 1
        return

    guard["routing_repair_optimizer_probe_count"] = int(
        guard.get("routing_repair_optimizer_probe_count", 0)
    ) + 1
    if evidence.get("update_applied") is True:
        guard["routing_repair_optimizer_update_applied_count"] = int(
            guard.get("routing_repair_optimizer_update_applied_count", 0)
        ) + 1
    clipped_abs_sum = _signature_value(evidence, "clipped_gradient", "abs_sum")
    if clipped_abs_sum > 0.0:
        guard["routing_repair_optimizer_nonzero_gradient_count"] = int(
            guard.get("routing_repair_optimizer_nonzero_gradient_count", 0)
        ) + 1

    sample = guard.setdefault("routing_repair_optimizer_probe_sample", [])
    if isinstance(sample, list) and len(sample) < MAX_ROUTING_REPAIR_OPTIMIZER_PROBE_SAMPLES:
        sample.append(
            {
                "learning_rate_scale": learning_rate_scale,
                "loss": loss,
                "update_applied": evidence.get("update_applied") is True,
                "learning_rate": float(evidence.get("learning_rate", 0.0) or 0.0),
                "raw_gradient_abs_sum": _signature_value(
                    evidence,
                    "raw_gradient",
                    "abs_sum",
                ),
                "clipped_gradient_abs_sum": clipped_abs_sum,
                "accumulated_gradient_available": _optional_available(
                    evidence,
                    "accumulated_gradient",
                ),
                "accumulated_gradient_abs_sum": _signature_value(
                    evidence,
                    "accumulated_gradient",
                    "abs_sum",
                ),
                "update_count_before": int(
                    evidence.get("update_count_before", 0) or 0
                ),
                "update_count_after": int(evidence.get("update_count_after", 0) or 0),
                "pending_accumulation_before": int(
                    evidence.get("pending_accumulation_before", 0) or 0
                ),
                "pending_accumulation_after": int(
                    evidence.get("pending_accumulation_after", 0) or 0
                ),
            }
        )


def _optional_available(evidence: dict[str, Any], key: str) -> bool:
    value = evidence.get(key)
    return isinstance(value, dict) and value.get("available") is True


def _signature_value(evidence: dict[str, Any], key: str, field: str) -> float:
    value = evidence.get(key)
    signature = value.get("signature", {}) if isinstance(value, dict) else {}
    try:
        return float(signature.get(field, 0.0))
    except (TypeError, ValueError):
        return 0.0
