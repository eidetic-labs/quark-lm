"""Compare candidate transformer training outputs to scalar fixtures."""

from __future__ import annotations

import math
from typing import Any

from transformer_backend_policy import (
    PYTORCH_BACKEND,
    validate_transformer_backend_metadata,
)
from transformer_training_parity_fixture import validate_training_parity_fixture
from transformer_training_parity_schema import (
    TRAINING_PARITY_REPORT_KIND,
    TRAINING_PARITY_SCHEMA_VERSION,
)


def build_training_parity_report(
    *,
    fixture: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare candidate training behavior against a scalar training fixture."""

    validate_training_parity_fixture(fixture)
    expected = fixture["training_case"]
    actual = candidate.get("training_case", {})
    tolerance = fixture["tolerance"]
    checks = [_backend_metadata_check(candidate.get("backend"))]
    checks.extend(
        [
            _number_check(
                "training_initial_loss",
                expected["initial_loss"],
                actual.get("initial_loss"),
                tolerance,
            ),
            _number_check(
                "training_final_loss",
                expected["final_loss"],
                actual.get("final_loss"),
                tolerance,
            ),
            _number_list_check(
                "training_final_logits",
                expected["final_logits"],
                actual.get("final_logits"),
                tolerance,
            ),
            _number_list_check(
                "training_step_losses",
                [record["loss"] for record in expected["step_records"]],
                [record.get("loss") for record in actual.get("step_records", [])],
                tolerance,
            ),
            _exact_check(
                "training_optimizer_state",
                expected["optimizer_state"],
                actual.get("optimizer_state"),
            ),
            _signature_check(
                expected["parameter_signature"],
                actual.get("parameter_signature"),
                tolerance,
            ),
        ]
    )
    return {
        "schema_version": TRAINING_PARITY_SCHEMA_VERSION,
        "kind": TRAINING_PARITY_REPORT_KIND,
        "fixture_id": fixture["fixture_id"],
        "candidate_backend": _candidate_backend_name(candidate),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "summary": _summary(checks),
    }


def _backend_metadata_check(backend: Any) -> dict[str, Any]:
    if not isinstance(backend, dict):
        return {
            "name": "backend_metadata",
            "passed": False,
            "error": "backend metadata is missing",
        }
    try:
        validate_transformer_backend_metadata(
            backend,
            require_artifact_fields=backend.get("backend") == PYTORCH_BACKEND,
        )
    except ValueError as exc:
        return {
            "name": "backend_metadata",
            "passed": False,
            "backend": backend.get("backend"),
            "error": str(exc),
        }
    return {
        "name": "backend_metadata",
        "passed": True,
        "backend": backend.get("backend"),
        "parity_status": backend.get("parity_status"),
    }


def _number_check(
    name: str,
    expected: float,
    actual: Any,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    if not isinstance(actual, int | float):
        return {"name": name, "passed": False, "error": "actual must be numeric"}
    actual_float = float(actual)
    return {
        "name": name,
        "passed": _close_enough(expected, actual_float, tolerance),
        "expected": expected,
        "actual": actual_float,
        "abs_diff": abs(expected - actual_float),
    }


def _number_list_check(
    name: str,
    expected: list[float],
    actual: Any,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return {
            "name": name,
            "passed": False,
            "error": "actual must be a list with matching length",
        }
    diffs = [abs(float(left) - float(right)) for left, right in zip(expected, actual)]
    return {
        "name": name,
        "passed": all(
            _close_enough(left, float(right), tolerance)
            for left, right in zip(expected, actual)
        ),
        "max_abs_diff": max(diffs, default=0.0),
    }


def _signature_check(
    expected: dict[str, Any],
    actual: Any,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    if not isinstance(actual, dict):
        return {"name": "training_parameter_signature", "passed": False}
    checks = [
        actual.get("count") == expected["count"],
        *[
            _close_enough(expected[key], float(actual.get(key, 0.0)), tolerance)
            for key in ("sum", "abs_sum", "square_sum")
        ],
    ]
    return {
        "name": "training_parameter_signature",
        "passed": all(checks),
        "expected": expected,
        "actual": actual,
    }


def _exact_check(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _close_enough(
    expected: float,
    actual: float,
    tolerance: dict[str, float],
) -> bool:
    return math.isclose(
        expected,
        actual,
        abs_tol=tolerance["absolute"],
        rel_tol=tolerance["relative"],
    )


def _candidate_backend_name(candidate: dict[str, Any]) -> str | None:
    backend = candidate.get("backend")
    return backend.get("backend") if isinstance(backend, dict) else None


def _summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "check_count": len(checks),
        "passed_check_count": sum(1 for check in checks if check["passed"]),
        "failed_checks": [
            check["name"] for check in checks if check["passed"] is not True
        ],
    }
