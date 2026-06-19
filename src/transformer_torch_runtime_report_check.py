"""Runtime-report consistency checks for PyTorch parity artifacts."""

from __future__ import annotations

from typing import Any

from transformer_torch_runtime_report import TORCH_RUNTIME_REPORT_KIND
from transformer_torch_runtime_report_validation import validate_torch_runtime_report


def build_torch_runtime_report_check(
    *,
    runtime_report: Any,
    runtime: Any,
    require_parity_attempt_allowed: bool | None = None,
    require_training_evidence_allowed: bool | None = None,
) -> dict[str, Any]:
    """Check that a PyTorch candidate carries consistent runtime evidence."""

    if not isinstance(runtime_report, dict):
        return _failed("runtime report is missing")
    try:
        validate_torch_runtime_report(runtime_report)
    except ValueError as exc:
        return _failed(str(exc))
    require_parity_attempt_allowed = _require_parity_attempt_allowed(
        require_parity_attempt_allowed=require_parity_attempt_allowed,
        require_training_evidence_allowed=require_training_evidence_allowed,
    )
    failures = _failures(
        runtime_report=runtime_report,
        runtime=runtime,
        require_parity_attempt_allowed=require_parity_attempt_allowed,
    )
    return {
        "name": "runtime_report",
        "passed": not failures,
        "status": runtime_report.get("status"),
        "evidence_scope": runtime_report.get("evidence_scope"),
        "parity_attempt_allowed": _parity_attempt_allowed(runtime_report),
        "training_evidence_allowed": runtime_report.get(
            "training_evidence_allowed"
        ),
        "failed_runtime_checks": failures,
    }


def _require_parity_attempt_allowed(
    *,
    require_parity_attempt_allowed: bool | None,
    require_training_evidence_allowed: bool | None,
) -> bool:
    if require_parity_attempt_allowed is not None:
        return require_parity_attempt_allowed
    return bool(require_training_evidence_allowed)


def _failures(
    *,
    runtime_report: dict[str, Any],
    runtime: Any,
    require_parity_attempt_allowed: bool,
) -> list[str]:
    failures = []
    if runtime_report.get("kind") != TORCH_RUNTIME_REPORT_KIND:
        failures.append("kind")
    if runtime_report.get("runtime") != runtime:
        failures.append("runtime")
    failures.extend(_boundary_failures(runtime_report.get("closed_world_boundary")))
    if (
        require_parity_attempt_allowed
        and _parity_attempt_allowed(runtime_report) is not True
    ):
        failures.append("parity_attempt_allowed")
    return failures


def _parity_attempt_allowed(runtime_report: dict[str, Any]) -> Any:
    if "parity_attempt_allowed" in runtime_report:
        return runtime_report.get("parity_attempt_allowed")
    return runtime_report.get("training_evidence_allowed")


def _boundary_failures(boundary: Any) -> list[str]:
    if not isinstance(boundary, dict):
        return ["closed_world_boundary"]
    expected = {
        "runtime_library_allowed": True,
        "learned_assets_imported": False,
        "training_data_imported": False,
        "pretrained_weights_imported": False,
        "pretrained_tokenizer_imported": False,
        "external_embeddings_imported": False,
    }
    return [
        key
        for key, expected_value in expected.items()
        if boundary.get(key) is not expected_value
    ]


def _failed(error: str) -> dict[str, Any]:
    return {
        "name": "runtime_report",
        "passed": False,
        "error": error,
    }
