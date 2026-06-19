"""Artifact-set validation for PyTorch training parity attempts."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_validation import (
    validate_torch_training_parity_attempt,
)


REQUIRED_TORCH_TRAINING_ATTEMPT_ARTIFACTS = (
    "attempt",
    "fixture",
    "candidate",
    "report",
)


def validate_torch_training_parity_attempt_artifact_set(
    artifacts: dict[str, Any],
    *,
    require_artifact_paths: bool = False,
) -> None:
    """Validate that a training parity attempt artifact set is coherent."""

    payloads = _required_payloads(artifacts)
    validate_torch_training_parity_attempt(
        payloads["attempt"],
        require_artifacts=require_artifact_paths,
    )
    _validate_fixture_ids(payloads)
    _validate_summary(
        "runtime",
        payloads["attempt"].get("runtime"),
        _runtime_summary(payloads["candidate"]),
    )
    _validate_summary(
        "candidate",
        payloads["attempt"].get("candidate"),
        _candidate_summary(payloads["candidate"]),
    )
    _validate_summary(
        "training_replay_parity_gate",
        payloads["attempt"].get("training_replay_parity_gate"),
        _gate_summary(payloads["candidate"].get("training_replay_parity_gate", {})),
    )
    _validate_summary(
        "training_parity_report",
        payloads["attempt"].get("training_parity_report"),
        _report_summary(payloads["report"]),
    )


def _required_payloads(artifacts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, dict):
        raise ValueError("training parity artifacts must be a dict")
    payloads = {}
    for key in REQUIRED_TORCH_TRAINING_ATTEMPT_ARTIFACTS:
        value = artifacts.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"artifacts.{key} must be a dict")
        payloads[key] = value
    return payloads


def _validate_fixture_ids(payloads: dict[str, dict[str, Any]]) -> None:
    fixture_id = payloads["attempt"].get("fixture_id")
    for key in ("fixture", "candidate", "report"):
        if payloads[key].get("fixture_id") != fixture_id:
            raise ValueError(f"artifacts.{key}.fixture_id is inconsistent")


def _validate_summary(name: str, actual: Any, expected: dict[str, Any]) -> None:
    if actual != expected:
        raise ValueError(f"attempt.{name} summary is inconsistent")


def _runtime_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    runtime_report = candidate.get("runtime_report", {})
    runtime = runtime_report.get("runtime", {})
    return {
        "status": runtime_report.get("status"),
        "passed": runtime_report.get("passed"),
        "parity_attempt_allowed": runtime_report.get("parity_attempt_allowed"),
        "runtime_kind": runtime.get("runtime_kind"),
        "device": runtime.get("device"),
        "dtype": runtime.get("dtype"),
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    backend = candidate.get("backend", {})
    return {
        "implementation_status": candidate.get("implementation_status"),
        "parity_status": backend.get("parity_status"),
        "training_readiness_status": candidate.get("training_readiness", {}).get(
            "status"
        ),
        "training_case_status": candidate.get("training_case", {}).get("status"),
    }


def _gate_summary(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": gate.get("status"),
        "passed": gate.get("passed"),
        "failed_checks": gate.get("summary", {}).get("failed_checks", []),
    }


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": report.get("passed"),
        "failed_checks": report.get("summary", {}).get("failed_checks", []),
    }
