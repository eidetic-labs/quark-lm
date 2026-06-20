"""Runtime-report validation for PyTorch training parity candidates."""

from __future__ import annotations

from typing import Any

from transformer_torch_runtime_report_check import build_torch_runtime_report_check


TORCH_TRAINING_CANDIDATE_RUNTIME_FIELDS = (
    "runtime",
    "runtime_report",
)


def validate_torch_training_candidate_runtime_report(
    candidate: dict[str, Any],
) -> None:
    """Validate that a candidate embeds runtime evidence for its runtime."""

    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a dict")
    runtime = _required_dict(candidate, "runtime")
    runtime_report = _required_dict(candidate, "runtime_report")
    check = build_torch_runtime_report_check(
        runtime_report=runtime_report,
        runtime=runtime,
    )
    if check.get("passed") is True:
        return
    failures = check.get("failed_runtime_checks") or [check.get("error", "invalid")]
    raise ValueError(f"candidate.runtime_report.{failures[0]} is inconsistent")


def _required_dict(candidate: dict[str, Any], key: str) -> dict[str, Any]:
    value = candidate.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"candidate.{key} must be a dict")
    return value
