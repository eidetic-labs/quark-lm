"""Standalone validation for PyTorch training backend promotion gates."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_attempt_boundary import (
    torch_training_attempt_boundary_failures,
)
from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION,
    TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
)


def validate_torch_training_backend_promotion_gate(
    gate: dict[str, Any],
    *,
    closed_world_boundary: dict[str, Any],
) -> None:
    """Validate the unpromoted backend-promotion gate contract."""

    if not isinstance(gate, dict):
        raise ValueError("training backend promotion gate must be a dict")
    if (
        gate.get("schema_version")
        != TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION
    ):
        raise ValueError("training backend promotion gate schema_version is invalid")
    if gate.get("status") != TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS:
        raise ValueError("training backend promotion gate status is invalid")
    if gate.get("passed") is not False:
        raise ValueError("training backend promotion gate must not pass")
    if gate.get("promotion_eligible") is not False:
        raise ValueError("training backend promotion gate must not be eligible")
    if gate.get("promoted_training_backend") is not False:
        raise ValueError("training backend promotion gate must not promote")
    if gate.get("evidence_scope") != "fixture_replay_parity_only":
        raise ValueError("training backend promotion gate evidence_scope is invalid")
    if not isinstance(gate.get("parity_evidence_matched"), bool):
        raise ValueError("training backend promotion gate parity status is invalid")
    _validate_checks(gate)
    _validate_parity_evidence(gate)
    _validate_future_gates(gate)
    _validate_boundary(gate, closed_world_boundary)


def _validate_checks(gate: dict[str, Any]) -> None:
    checks = gate.get("checks")
    if not isinstance(checks, list):
        raise ValueError("training backend promotion gate checks are invalid")
    if [check.get("name") for check in checks] != list(
        TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS
    ):
        raise ValueError("training backend promotion gate check catalog is invalid")
    for check in checks:
        if not isinstance(check.get("passed"), bool):
            raise ValueError("training backend promotion gate check status is invalid")
        if not isinstance(check.get("reason"), str) or not check["reason"].strip():
            raise ValueError("training backend promotion gate check reason is invalid")
    expected_blockers = [
        check["name"]
        for check in checks
        if check.get("passed") is False
    ]
    if gate.get("blockers") != expected_blockers:
        raise ValueError("training backend promotion gate blockers are invalid")


def _validate_parity_evidence(gate: dict[str, Any]) -> None:
    check_status = {check["name"]: check["passed"] for check in gate["checks"]}
    if (
        gate["parity_evidence_matched"] is True
        and check_status.get("training_parity_report") is not True
    ):
        raise ValueError("training backend promotion gate parity evidence is invalid")


def _validate_future_gates(gate: dict[str, Any]) -> None:
    if gate.get("required_future_gates") != list(
        TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES
    ):
        raise ValueError("training backend promotion gate future gates are invalid")


def _validate_boundary(
    gate: dict[str, Any],
    closed_world_boundary: dict[str, Any],
) -> None:
    boundary_failures = torch_training_attempt_boundary_failures(
        closed_world_boundary
    )
    if gate.get("closed_world_boundary_passed") is not (not boundary_failures):
        raise ValueError("training backend promotion gate boundary status is invalid")
    if gate.get("closed_world_boundary_failures") != boundary_failures:
        raise ValueError("training backend promotion gate boundary failures are invalid")
