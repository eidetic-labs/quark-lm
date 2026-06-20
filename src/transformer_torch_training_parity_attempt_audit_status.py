"""Status consistency validation for compact PyTorch attempt audits."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_status import (
    resolve_torch_training_parity_attempt_passed,
    resolve_torch_training_parity_attempt_status,
)


def validate_torch_training_parity_attempt_audit_status(
    audit: dict[str, Any],
) -> None:
    """Validate compact audit attempt status against replay/report evidence."""

    expected_status = resolve_torch_training_parity_attempt_status(
        runtime={
            "status": audit.get("runtime_status"),
            "parity_attempt_allowed": audit.get("parity_attempt_allowed"),
        },
        training_replay_parity_gate={
            "status": audit.get("training_replay_parity_status"),
            "passed": audit.get("training_replay_parity_passed"),
        },
        training_parity_report={
            "passed": audit.get("training_report_passed"),
        },
    )
    if audit.get("attempt_status") != expected_status:
        raise ValueError("audit.attempt_status is inconsistent")
    expected_passed = resolve_torch_training_parity_attempt_passed(
        runtime={
            "status": audit.get("runtime_status"),
            "parity_attempt_allowed": audit.get("parity_attempt_allowed"),
        },
        training_replay_parity_gate={
            "status": audit.get("training_replay_parity_status"),
            "passed": audit.get("training_replay_parity_passed"),
        },
        training_parity_report={
            "passed": audit.get("training_report_passed"),
        },
    )
    if audit.get("attempt_passed") is not expected_passed:
        raise ValueError("audit.attempt_passed is inconsistent")
