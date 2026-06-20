"""Standalone validation for optional PyTorch runtime preflight reports."""

from __future__ import annotations

from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from transformer_torch_runtime import (
    TORCH_RUNTIME_KIND_PYTORCH,
    TORCH_RUNTIME_KIND_TEST_DOUBLE,
    TORCH_RUNTIME_KIND_UNAVAILABLE,
)
from transformer_torch_runtime_report import TORCH_RUNTIME_REPORT_KIND


TORCH_RUNTIME_REPORT_STATUSES = (
    "ready_for_pytorch_parity",
    "blocked_runtime_unavailable",
    "blocked_test_double_runtime",
    "blocked_dtype_unavailable",
    "blocked_pytorch_runtime",
)
TORCH_RUNTIME_REPORT_CHECKS = (
    "runtime_available",
    "runtime_kind",
    "dtype_available",
)
TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE = "runtime_preflight_only"


def validate_torch_runtime_report(report: dict[str, Any]) -> None:
    """Validate runtime preflight evidence before it gates PyTorch attempts."""

    if not isinstance(report, dict):
        raise ValueError("runtime_report must be a dict")
    if set(report) != _REPORT_KEYS:
        raise ValueError("runtime_report keys are inconsistent")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("runtime_report.schema_version is inconsistent")
    if report.get("kind") != TORCH_RUNTIME_REPORT_KIND:
        raise ValueError("runtime_report.kind is inconsistent")
    status = _require_non_empty_string(report, "status")
    if status not in TORCH_RUNTIME_REPORT_STATUSES:
        raise ValueError("runtime_report.status is unsupported")
    _require_bool(report, "passed")
    _require_dict(report, "runtime")
    _validate_runtime(report["runtime"])
    _validate_checks(report)
    _validate_summary(report)
    _validate_boundary(report.get("closed_world_boundary"))
    _validate_flags(report)
    _validate_status(report)
    _require_non_empty_string(report, "reason")


def _validate_runtime(runtime: dict[str, Any]) -> None:
    if set(runtime) != _runtime_keys(runtime):
        raise ValueError("runtime_report.runtime keys are inconsistent")
    _require_bool(runtime, "available", label="runtime")
    _require_bool(runtime, "dtype_available", label="runtime")
    runtime_kind = _require_non_empty_string(
        runtime,
        "runtime_kind",
        label="runtime",
    )
    if runtime_kind not in _RUNTIME_KINDS:
        raise ValueError("runtime_report.runtime.runtime_kind is unsupported")
    _require_non_empty_string(runtime, "device", label="runtime")
    _require_non_empty_string(runtime, "dtype", label="runtime")


def _validate_checks(report: dict[str, Any]) -> None:
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise ValueError("runtime_report.checks must be a list")
    if [check.get("name") for check in checks] != list(TORCH_RUNTIME_REPORT_CHECKS):
        raise ValueError("runtime_report.check catalog is inconsistent")
    expected = _expected_check_results(report["runtime"])
    for check in checks:
        _validate_check_keys(check)
        name = check["name"]
        if not isinstance(check.get("passed"), bool):
            raise ValueError(f"runtime_report.checks.{name}.passed is invalid")
        if check["passed"] is not expected[name]:
            raise ValueError(f"runtime_report.checks.{name}.passed is inconsistent")


def _validate_summary(report: dict[str, Any]) -> None:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("runtime_report.summary must be a dict")
    if set(summary) != _SUMMARY_KEYS:
        raise ValueError("runtime_report.summary keys are inconsistent")
    checks = report["checks"]
    failed = [check["name"] for check in checks if check.get("passed") is not True]
    if summary.get("check_count") != len(checks):
        raise ValueError("runtime_report.summary.check_count is inconsistent")
    if summary.get("passed_check_count") != len(checks) - len(failed):
        raise ValueError("runtime_report.summary.passed_check_count is inconsistent")
    if summary.get("failed_checks") != failed:
        raise ValueError("runtime_report.summary.failed_checks is inconsistent")


def _validate_boundary(boundary: Any) -> None:
    if not isinstance(boundary, dict):
        raise ValueError("runtime_report.closed_world_boundary must be a dict")
    if set(boundary) != _BOUNDARY_KEYS:
        raise ValueError("runtime_report.closed_world_boundary keys are inconsistent")
    expected = {
        "runtime_library_allowed": True,
        "learned_assets_imported": False,
        "training_data_imported": False,
        "pretrained_weights_imported": False,
        "pretrained_tokenizer_imported": False,
        "external_embeddings_imported": False,
    }
    for key, expected_value in expected.items():
        if boundary.get(key) is not expected_value:
            raise ValueError(f"runtime_report.closed_world_boundary.{key}")


def _runtime_keys(runtime: dict[str, Any]) -> set[str]:
    if runtime.get("available") is True:
        return _AVAILABLE_RUNTIME_KEYS
    return _UNAVAILABLE_RUNTIME_KEYS


def _validate_check_keys(check: dict[str, Any]) -> None:
    expected = (
        _STATUS_CHECK_KEYS
        if check.get("name") == "runtime_kind"
        else _BOOL_CHECK_KEYS
    )
    if set(check) != expected:
        raise ValueError(f"runtime_report.checks.{check.get('name')}.keys")


def _validate_flags(report: dict[str, Any]) -> None:
    if report.get("evidence_scope") != TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE:
        raise ValueError("runtime_report.evidence_scope is inconsistent")
    _require_bool(report, "parity_attempt_allowed")
    _require_bool(report, "training_evidence_allowed")
    if report["parity_attempt_allowed"] is not report["passed"]:
        raise ValueError("runtime_report.parity_attempt_allowed is inconsistent")
    if report["training_evidence_allowed"] is not report["passed"]:
        raise ValueError("runtime_report.training_evidence_allowed is inconsistent")


def _validate_status(report: dict[str, Any]) -> None:
    passed = all(check["passed"] for check in report["checks"])
    if report["passed"] is not passed:
        raise ValueError("runtime_report.passed is inconsistent")
    if report["status"] != _expected_status(report["runtime"], passed):
        raise ValueError("runtime_report.status is inconsistent")


def _expected_status(runtime: dict[str, Any], passed: bool) -> str:
    if passed:
        return "ready_for_pytorch_parity"
    if not runtime.get("available"):
        return "blocked_runtime_unavailable"
    if runtime.get("runtime_kind") == TORCH_RUNTIME_KIND_TEST_DOUBLE:
        return "blocked_test_double_runtime"
    if not runtime.get("dtype_available"):
        return "blocked_dtype_unavailable"
    return "blocked_pytorch_runtime"


def _expected_check_results(runtime: dict[str, Any]) -> dict[str, bool]:
    return {
        "runtime_available": bool(runtime.get("available")),
        "runtime_kind": runtime.get("runtime_kind") == TORCH_RUNTIME_KIND_PYTORCH,
        "dtype_available": bool(runtime.get("dtype_available")),
    }


def _require_dict(record: dict[str, Any], key: str) -> None:
    if not isinstance(record.get(key), dict):
        raise ValueError(f"runtime_report.{key} must be a dict")


def _require_bool(
    record: dict[str, Any],
    key: str,
    *,
    label: str = "runtime_report",
) -> None:
    if not isinstance(record.get(key), bool):
        raise ValueError(f"{label}.{key} must be a bool")


def _require_non_empty_string(
    record: dict[str, Any],
    key: str,
    *,
    label: str = "runtime_report",
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


_RUNTIME_KINDS = (
    TORCH_RUNTIME_KIND_PYTORCH,
    TORCH_RUNTIME_KIND_TEST_DOUBLE,
    TORCH_RUNTIME_KIND_UNAVAILABLE,
)
_REPORT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "passed",
    "runtime",
    "checks",
    "summary",
    "evidence_scope",
    "parity_attempt_allowed",
    "closed_world_boundary",
    "training_evidence_allowed",
    "reason",
}
_UNAVAILABLE_RUNTIME_KEYS = {
    "backend",
    "available",
    "runtime_kind",
    "version",
    "requested_device",
    "device",
    "requested_dtype",
    "dtype",
    "dtype_available",
    "error",
}
_AVAILABLE_RUNTIME_KEYS = _UNAVAILABLE_RUNTIME_KEYS | {"available_devices"}
_SUMMARY_KEYS = {
    "check_count",
    "passed_check_count",
    "failed_checks",
}
_BOUNDARY_KEYS = {
    "runtime_library_allowed",
    "learned_assets_imported",
    "training_data_imported",
    "pretrained_weights_imported",
    "pretrained_tokenizer_imported",
    "external_embeddings_imported",
}
_BOOL_CHECK_KEYS = {"name", "passed", "actual"}
_STATUS_CHECK_KEYS = {"name", "passed", "expected", "actual"}
