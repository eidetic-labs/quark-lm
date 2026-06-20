"""Next-requirements consistency for compact PyTorch attempt audits."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_compact_requirements import (
    build_torch_training_parity_attempt_compact_requirements,
)


def validate_torch_training_parity_attempt_audit_requirements(
    audit: dict[str, Any],
) -> None:
    """Validate compact audit next-requirements against compact evidence."""

    expected = build_torch_training_parity_attempt_compact_requirements(
        runtime={
            "status": audit.get("runtime_status"),
            "passed": audit.get("parity_attempt_allowed"),
            "parity_attempt_allowed": audit.get("parity_attempt_allowed"),
            "failed_checks": list(audit.get("runtime_failed_checks", [])),
        },
        candidate={
            "training_readiness_status": audit.get("training_readiness_status"),
            "training_readiness_failed_checks": list(
                audit.get("training_readiness_failed_checks", [])
            ),
        },
        training_replay_parity_gate={
            "status": audit.get("training_replay_parity_status"),
            "passed": audit.get("training_replay_parity_passed"),
            "failed_checks": list(
                audit.get("training_replay_parity_failed_checks", [])
            ),
        },
        training_parity_report={
            "passed": audit.get("training_report_passed"),
            "failed_checks": list(audit.get("training_report_failed_checks", [])),
        },
    )
    _compare(audit, "next_requirements_stage", expected["stage"])
    _compare(audit, "next_requirements_status", expected["status"])
    _compare(audit, "next_actions", expected["next_actions"])


def _compare(audit: dict[str, Any], key: str, expected: Any) -> None:
    if audit.get(key) != expected:
        raise ValueError(f"audit.{key} is inconsistent")
