"""Parameter-signature comparison for PyTorch training parity probes."""

from __future__ import annotations

import math
from typing import Any


TORCH_PARAMETER_SIGNATURE_COMPARISON_SCHEMA_VERSION = 1
TORCH_PARAMETER_SIGNATURE_MATCHED_STATUS = "parameter_signature_matched"
TORCH_PARAMETER_SIGNATURE_MISMATCH_STATUS = "parameter_signature_mismatch"


def build_torch_parameter_signature_comparison(
    *,
    expected_signature: dict[str, Any],
    actual_signature: dict[str, Any],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    """Compare a candidate parameter signature against scalar evidence."""

    checks = [
        _count_check(expected_signature, actual_signature),
        *[
            _number_check(name, expected_signature, actual_signature, tolerance)
            for name in ("sum", "abs_sum", "square_sum")
        ],
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": TORCH_PARAMETER_SIGNATURE_COMPARISON_SCHEMA_VERSION,
        "status": (
            TORCH_PARAMETER_SIGNATURE_MATCHED_STATUS
            if passed
            else TORCH_PARAMETER_SIGNATURE_MISMATCH_STATUS
        ),
        "passed": passed,
        "expected_signature": expected_signature,
        "actual_signature": actual_signature,
        "checks": checks,
        "max_abs_diff": max(
            (
                check["abs_diff"]
                for check in checks
                if "abs_diff" in check
            ),
            default=0.0,
        ),
        "failed_checks": [
            check["name"] for check in checks if check["passed"] is not True
        ],
    }


def _count_check(
    expected_signature: dict[str, Any],
    actual_signature: dict[str, Any],
) -> dict[str, Any]:
    expected = expected_signature["count"]
    actual = actual_signature.get("count")
    return {
        "name": "count",
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _number_check(
    name: str,
    expected_signature: dict[str, Any],
    actual_signature: dict[str, Any],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    expected = float(expected_signature[name])
    actual = float(actual_signature.get(name, 0.0))
    return {
        "name": name,
        "passed": math.isclose(
            expected,
            actual,
            abs_tol=tolerance["absolute"],
            rel_tol=tolerance["relative"],
        ),
        "expected": expected,
        "actual": actual,
        "abs_diff": abs(expected - actual),
    }
