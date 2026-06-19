"""Compare replayed PyTorch gradients with scalar optimizer evidence."""

from __future__ import annotations

import math
from typing import Any


TORCH_REPLAY_GRADIENT_COMPARISON_SCHEMA_VERSION = 1
TORCH_REPLAY_GRADIENT_MATCHED_STATUS = "replay_gradient_signature_matched"
TORCH_REPLAY_GRADIENT_MISMATCH_STATUS = "replay_gradient_signature_mismatch"


def build_torch_replay_gradient_comparison(
    *,
    scalar_step_record: dict[str, Any],
    torch_gradient_snapshot: dict[str, Any],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    """Compare clipped PyTorch microstep gradients to scalar evidence."""

    scalar_evidence = scalar_step_record["optimizer_gradient_evidence"]
    scalar_signature = scalar_evidence["clipped_gradient"]["signature"]
    actual_signature = torch_gradient_snapshot["signature"]
    matched = _signatures_match(
        expected=scalar_signature,
        actual=actual_signature,
        tolerance=tolerance,
    )
    return {
        "schema_version": TORCH_REPLAY_GRADIENT_COMPARISON_SCHEMA_VERSION,
        "status": (
            TORCH_REPLAY_GRADIENT_MATCHED_STATUS
            if matched
            else TORCH_REPLAY_GRADIENT_MISMATCH_STATUS
        ),
        "passed": matched,
        "step": scalar_step_record["step"],
        "compared_source": "scalar_clipped_gradient",
        "expected_signature": scalar_signature,
        "actual_signature": actual_signature,
        "signature_abs_diff": _signature_abs_diff(
            expected=scalar_signature,
            actual=actual_signature,
        ),
        "scalar_buffer_after_add_signature": scalar_evidence[
            "buffer_after_add"
        ]["signature"],
        "scalar_accumulated_gradient_signature": scalar_evidence[
            "accumulated_gradient"
        ]["signature"],
        "buffer_parity_proven": False,
        "reason": _reason(matched),
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
        "square_sum": abs(
            float(actual["square_sum"]) - float(expected["square_sum"])
        ),
    }


def _reason(matched: bool) -> str:
    if matched:
        return "replayed clipped gradient signature matches scalar evidence"
    return "replayed clipped gradient signature does not match scalar evidence"
