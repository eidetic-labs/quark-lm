"""Compare candidate transformer backend outputs to scalar parity fixtures."""

from __future__ import annotations

import math
from typing import Any

from transformer_backend_parity_schema import PARITY_REPORT_KIND, PARITY_SCHEMA_VERSION
from transformer_backend_parity_validation import validate_backend_parity_fixture
from transformer_backend_policy import (
    PYTORCH_BACKEND,
    validate_transformer_backend_metadata,
)


def build_backend_parity_report(
    *,
    fixture: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare candidate backend outputs against a scalar parity fixture."""

    validate_backend_parity_fixture(fixture)
    checks = [_backend_metadata_check(candidate.get("backend"))]
    checks.extend(
        _forward_case_checks(
            fixture["forward_cases"],
            candidate.get("forward_cases", []),
            fixture["tolerance"],
        )
    )
    checks.extend(
        _generation_case_checks(
            fixture.get("generation_cases", []),
            candidate.get("generation_cases", []),
        )
    )
    return {
        "schema_version": PARITY_SCHEMA_VERSION,
        "kind": PARITY_REPORT_KIND,
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


def _forward_case_checks(
    expected_cases: list[dict[str, Any]],
    candidate_cases: Any,
    tolerance: dict[str, float],
) -> list[dict[str, Any]]:
    candidates = _case_map(candidate_cases)
    checks: list[dict[str, Any]] = []
    for expected in expected_cases:
        case_id = expected["case_id"]
        candidate = candidates.get(case_id)
        if candidate is None:
            checks.append(_missing_check(f"forward:{case_id}"))
            continue
        checks.append(
            _number_list_check(
                f"forward_logits:{case_id}",
                expected["logits"],
                candidate.get("logits"),
                tolerance,
            )
        )
        checks.append(
            _number_check(
                f"forward_loss:{case_id}",
                expected["loss"],
                candidate.get("loss"),
                tolerance,
            )
        )
    return checks


def _generation_case_checks(
    expected_cases: list[dict[str, Any]],
    candidate_cases: Any,
) -> list[dict[str, Any]]:
    candidates = _case_map(candidate_cases)
    checks: list[dict[str, Any]] = []
    for expected in expected_cases:
        case_id = expected["case_id"]
        candidate = candidates.get(case_id)
        if candidate is None:
            checks.append(_missing_check(f"generation:{case_id}"))
            continue
        checks.extend(
            [
                _exact_check(
                    f"generation_text:{case_id}",
                    expected["text"],
                    candidate.get("text"),
                ),
                _exact_check(
                    f"generation_tokens:{case_id}",
                    expected["token_ids"],
                    candidate.get("token_ids"),
                ),
            ]
        )
    return checks


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
            "error": "actual value must be a list with matching length",
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


def _number_check(
    name: str,
    expected: float,
    actual: Any,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    if not isinstance(actual, int | float):
        return {"name": name, "passed": False, "error": "actual value must be numeric"}
    actual_float = float(actual)
    return {
        "name": name,
        "passed": _close_enough(expected, actual_float, tolerance),
        "abs_diff": abs(expected - actual_float),
        "expected": expected,
        "actual": actual_float,
    }


def _exact_check(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _case_map(cases: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(cases, list):
        return {}
    return {
        case["case_id"]: case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
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


def _missing_check(name: str) -> dict[str, Any]:
    return {"name": name, "passed": False, "error": "candidate case is missing"}
