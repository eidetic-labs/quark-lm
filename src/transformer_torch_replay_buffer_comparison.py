"""Compare replayed PyTorch gradient buffers with scalar evidence."""

from __future__ import annotations

import math
from typing import Any


TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION = 1
TORCH_REPLAY_BUFFER_MATCHED_STATUS = "replay_buffer_signature_matched"
TORCH_REPLAY_BUFFER_MISMATCH_STATUS = "replay_buffer_signature_mismatch"
TORCH_REPLAY_BUFFER_NOT_RUN_STATUS = "replay_buffer_comparison_not_run"


def build_torch_replay_buffer_comparison(
    *,
    fixture: dict[str, Any],
    replay_control_probe: dict[str, Any],
) -> dict[str, Any]:
    """Fold replayed gradients into a buffer and compare scalar evidence."""

    if replay_control_probe.get("status") != "accumulation_replay_control_recorded":
        return _not_run("replay control did not complete")

    parameter_count = fixture["parameter_manifest"]["parameter_count"]
    buffer = [0.0 for _index in range(parameter_count)]
    records = []
    for microstep, scalar_record in zip(
        replay_control_probe["microsteps"],
        fixture["training_case"]["step_records"],
    ):
        record = _comparison_record(
            buffer_before=buffer,
            microstep=microstep,
            scalar_record=scalar_record,
            tolerance=fixture["tolerance"],
        )
        records.append(record)
        buffer = (
            [0.0 for _index in range(parameter_count)]
            if record["update_applied"]
            else record["actual_buffer_after_add_values"]
        )

    passed = records and all(record["passed"] for record in records)
    return {
        "schema_version": TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
        "status": (
            TORCH_REPLAY_BUFFER_MATCHED_STATUS
            if passed
            else TORCH_REPLAY_BUFFER_MISMATCH_STATUS
        ),
        "passed": passed,
        "case_id": replay_control_probe["case_id"],
        "step_count": len(records),
        "matched_step_count": sum(1 for record in records if record["passed"]),
        "mismatched_step_count": sum(
            1 for record in records if not record["passed"]
        ),
        "update_step_count": sum(1 for record in records if record["update_applied"]),
        "buffered_gradient_parity_proven": passed,
        "optimizer_update_parity_proven": False,
        "final_loss_parity_proven": False,
        "reason": (
            "replayed gradient buffers match scalar evidence"
            if passed
            else "replayed gradient buffers do not match scalar evidence"
        ),
        "records": [_public_record(record) for record in records],
    }


def _comparison_record(
    *,
    buffer_before: list[float],
    microstep: dict[str, Any],
    scalar_record: dict[str, Any],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    evidence = scalar_record["optimizer_gradient_evidence"]
    gradient_values = _snapshot_values(microstep["gradient_snapshot"])
    buffer_after_add = _add_vectors(buffer_before, gradient_values)
    accumulated_values = _accumulated_values(
        buffer_after_add=buffer_after_add,
        evidence=evidence,
    )
    comparisons = {
        "buffer_before": _signature_comparison(
            expected=evidence["buffer_before"]["signature"],
            actual=_signature(buffer_before),
            tolerance=tolerance,
        ),
        "buffer_after_add": _signature_comparison(
            expected=evidence["buffer_after_add"]["signature"],
            actual=_signature(buffer_after_add),
            tolerance=tolerance,
        ),
        "accumulated_gradient": _accumulated_comparison(
            evidence=evidence,
            actual_values=accumulated_values,
            tolerance=tolerance,
        ),
    }
    return {
        "step": scalar_record["step"],
        "update_applied": evidence["update_applied"],
        "actual_buffer_after_add_values": buffer_after_add,
        "comparisons": comparisons,
        "passed": all(comparison["passed"] for comparison in comparisons.values()),
    }


def _snapshot_values(snapshot: dict[str, Any]) -> list[float]:
    return [
        value
        for parameter in snapshot["parameters"]
        for value in parameter["values"]
    ]


def _add_vectors(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        return list(right)
    return [left_value + right_value for left_value, right_value in zip(left, right)]


def _accumulated_values(
    *,
    buffer_after_add: list[float],
    evidence: dict[str, Any],
) -> list[float]:
    if not evidence["accumulated_gradient"]["available"]:
        return []
    divisor = evidence["pending_accumulation_before"] + 1
    return [value / divisor for value in buffer_after_add]


def _accumulated_comparison(
    *,
    evidence: dict[str, Any],
    actual_values: list[float],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    expected = evidence["accumulated_gradient"]
    if not expected["available"]:
        return {
            "status": "not_applicable",
            "passed": not actual_values,
            "expected_available": False,
            "actual_available": bool(actual_values),
            "expected_signature": expected["signature"],
            "actual_signature": _signature(actual_values),
            "signature_abs_diff": _signature_abs_diff(
                expected=expected["signature"],
                actual=_signature(actual_values),
            ),
        }
    comparison = _signature_comparison(
        expected=expected["signature"],
        actual=_signature(actual_values),
        tolerance=tolerance,
    )
    return {
        **comparison,
        "expected_available": True,
        "actual_available": bool(actual_values),
    }


def _signature_comparison(
    *,
    expected: dict[str, float | int],
    actual: dict[str, float | int],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    passed = _signatures_match(
        expected=expected,
        actual=actual,
        tolerance=tolerance,
    )
    return {
        "status": "matched" if passed else "mismatch",
        "passed": passed,
        "expected_signature": expected,
        "actual_signature": actual,
        "signature_abs_diff": _signature_abs_diff(expected=expected, actual=actual),
    }


def _signatures_match(
    *,
    expected: dict[str, float | int],
    actual: dict[str, float | int],
    tolerance: dict[str, float],
) -> bool:
    if expected["count"] != actual["count"]:
        return False
    return all(
        math.isclose(
            float(expected[key]),
            float(actual[key]),
            abs_tol=tolerance["absolute"],
            rel_tol=tolerance["relative"],
        )
        for key in ("sum", "abs_sum", "square_sum")
    )


def _signature_abs_diff(
    *,
    expected: dict[str, float | int],
    actual: dict[str, float | int],
) -> dict[str, float | int]:
    return {
        "count": int(actual["count"]) - int(expected["count"]),
        "sum": abs(float(actual["sum"]) - float(expected["sum"])),
        "abs_sum": abs(float(actual["abs_sum"]) - float(expected["abs_sum"])),
        "square_sum": abs(float(actual["square_sum"]) - float(expected["square_sum"])),
    }


def _signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": record["step"],
        "update_applied": record["update_applied"],
        "passed": record["passed"],
        "comparisons": record["comparisons"],
    }


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
        "status": TORCH_REPLAY_BUFFER_NOT_RUN_STATUS,
        "passed": False,
        "reason": reason,
        "buffered_gradient_parity_proven": False,
        "optimizer_update_parity_proven": False,
        "final_loss_parity_proven": False,
    }
