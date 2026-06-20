"""Build next-requirements artifacts from compact PyTorch attempt summaries."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_requirements import (
    build_torch_training_parity_attempt_requirements,
)


def build_torch_training_parity_attempt_compact_requirements(
    *,
    runtime: dict[str, Any],
    candidate: dict[str, Any],
    training_replay_parity_gate: dict[str, Any],
    training_parity_report: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild next-requirements routing from persisted compact summaries."""

    return build_torch_training_parity_attempt_requirements(
        runtime_report={
            "status": runtime.get("status"),
            "parity_attempt_allowed": runtime.get("parity_attempt_allowed"),
            "summary": {"failed_checks": list(runtime.get("failed_checks", []))},
        },
        candidate={
            "training_readiness": {
                "status": candidate.get("training_readiness_status"),
                "summary": {
                    "failed_checks": list(
                        candidate.get("training_readiness_failed_checks", [])
                    ),
                },
            },
            "training_replay_parity_gate": {
                "status": training_replay_parity_gate.get("status"),
                "passed": training_replay_parity_gate.get("passed"),
                "summary": {
                    "failed_checks": list(
                        training_replay_parity_gate.get("failed_checks", [])
                    ),
                },
            },
        },
        report={
            "passed": training_parity_report.get("passed"),
            "summary": {
                "failed_checks": list(training_parity_report.get("failed_checks", [])),
            },
        },
    )
