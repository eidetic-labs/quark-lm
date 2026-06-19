"""Signature helpers for PyTorch replay evidence comparisons."""

from __future__ import annotations

import math
from typing import Any


def build_value_signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


def compare_value_signatures(
    *,
    expected: dict[str, float | int],
    actual: dict[str, float | int],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    passed = value_signatures_match(
        expected=expected,
        actual=actual,
        tolerance=tolerance,
    )
    return {
        "status": "matched" if passed else "mismatch",
        "passed": passed,
        "expected_signature": expected,
        "actual_signature": actual,
        "signature_abs_diff": value_signature_abs_diff(
            expected=expected,
            actual=actual,
        ),
    }


def value_signatures_match(
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


def value_signature_abs_diff(
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
