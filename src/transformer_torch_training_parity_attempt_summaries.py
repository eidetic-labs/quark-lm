"""Compact summary builders for PyTorch training parity attempts."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_hashes import (
    build_torch_runtime_report_hash,
    build_torch_training_attempt_payload_hash,
)


def build_torch_attempt_runtime_summary(
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize runtime preflight evidence without losing payload identity."""

    runtime = runtime_report.get("runtime", {})
    return {
        "status": runtime_report.get("status"),
        "passed": runtime_report.get("passed"),
        "parity_attempt_allowed": runtime_report.get("parity_attempt_allowed"),
        "runtime_kind": runtime.get("runtime_kind"),
        "device": runtime.get("device"),
        "dtype": runtime.get("dtype"),
        "runtime_report_sha256": build_torch_runtime_report_hash(runtime_report),
    }


def build_torch_attempt_candidate_summary(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Summarize candidate routing evidence with a full-payload hash."""

    backend = candidate.get("backend", {})
    return {
        "implementation_status": candidate.get("implementation_status"),
        "parity_status": backend.get("parity_status"),
        "training_readiness_status": candidate.get("training_readiness", {}).get(
            "status"
        ),
        "training_case_status": candidate.get("training_case", {}).get("status"),
        "candidate_sha256": build_torch_training_attempt_payload_hash(
            candidate,
            payload_name="candidate",
        ),
    }


def build_torch_attempt_replay_gate_summary(
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Summarize replay-gate evidence with a full-payload hash."""

    return {
        "status": gate.get("status"),
        "passed": gate.get("passed"),
        "failed_checks": gate.get("summary", {}).get("failed_checks", []),
        "training_replay_parity_gate_sha256": (
            build_torch_training_attempt_payload_hash(
                gate,
                payload_name="training_replay_parity_gate",
            )
        ),
    }


def build_torch_attempt_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize training parity report evidence with a full-payload hash."""

    return {
        "passed": report.get("passed"),
        "failed_checks": report.get("summary", {}).get("failed_checks", []),
        "training_parity_report_sha256": build_torch_training_attempt_payload_hash(
            report,
            payload_name="training_parity_report",
        ),
    }
