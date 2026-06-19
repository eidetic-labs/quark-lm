"""Runtime-report consistency checks for PyTorch parity artifacts."""

from __future__ import annotations

from typing import Any

from transformer_torch_runtime_report import TORCH_RUNTIME_REPORT_KIND


def build_torch_runtime_report_check(
    *,
    runtime_report: Any,
    runtime: Any,
    require_training_evidence_allowed: bool,
) -> dict[str, Any]:
    """Check that a PyTorch candidate carries consistent runtime evidence."""

    if not isinstance(runtime_report, dict):
        return _failed("runtime report is missing")
    failures = _failures(
        runtime_report=runtime_report,
        runtime=runtime,
        require_training_evidence_allowed=require_training_evidence_allowed,
    )
    return {
        "name": "runtime_report",
        "passed": not failures,
        "status": runtime_report.get("status"),
        "training_evidence_allowed": runtime_report.get(
            "training_evidence_allowed"
        ),
        "failed_runtime_checks": failures,
    }


def _failures(
    *,
    runtime_report: dict[str, Any],
    runtime: Any,
    require_training_evidence_allowed: bool,
) -> list[str]:
    failures = []
    if runtime_report.get("kind") != TORCH_RUNTIME_REPORT_KIND:
        failures.append("kind")
    if runtime_report.get("runtime") != runtime:
        failures.append("runtime")
    failures.extend(_boundary_failures(runtime_report.get("closed_world_boundary")))
    if (
        require_training_evidence_allowed
        and runtime_report.get("training_evidence_allowed") is not True
    ):
        failures.append("training_evidence_allowed")
    return failures


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
