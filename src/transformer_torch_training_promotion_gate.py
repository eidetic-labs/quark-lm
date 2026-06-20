"""Promotion boundary for optional PyTorch training parity evidence."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_attempt_boundary import (
    torch_training_attempt_boundary_failures,
)
from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
)


TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION = 1
TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS = "training_backend_not_promoted"
TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS = (
    "training_parity_report",
    "closed_world_boundary",
    "fixture_scope_only",
    "model_quality_gate",
)
TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES = (
    "general_training_backend_gate",
    "model_quality_eval_gate",
    "profile_and_retention_gate",
)


def build_torch_training_backend_promotion_gate(
    *,
    candidate: dict[str, Any],
    report: dict[str, Any],
    closed_world_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Record why replay parity evidence is not backend promotion evidence."""

    boundary_failures = _boundary_failures(closed_world_boundary)
    checks = [
        _check(
            "training_parity_report",
            report.get("passed") is True,
            "training parity report must pass before promotion can be considered",
        ),
        _check(
            "closed_world_boundary",
            not boundary_failures,
            "closed-world boundary flags must remain clean",
        ),
        _check(
            "fixture_scope_only",
            False,
            "current evidence covers a tiny parity fixture, not a general trainer",
        ),
        _check(
            "model_quality_gate",
            False,
            "model-quality eval and profile gates have not promoted PyTorch",
        ),
    ]
    return {
        "schema_version": TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION,
        "status": TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
        "passed": False,
        "promotion_eligible": False,
        "promoted_training_backend": False,
        "evidence_scope": "fixture_replay_parity_only",
        "parity_evidence_matched": _parity_evidence_matched(
            candidate=candidate,
            report=report,
        ),
        "closed_world_boundary_passed": not boundary_failures,
        "closed_world_boundary_failures": boundary_failures,
        "checks": checks,
        "blockers": [check["name"] for check in checks if not check["passed"]],
        "required_future_gates": list(
            TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES
        ),
    }


def _parity_evidence_matched(
    *,
    candidate: dict[str, Any],
    report: dict[str, Any],
) -> bool:
    replay_gate = candidate.get("training_replay_parity_gate", {})
    backend = candidate.get("backend", {})
    return (
        report.get("passed") is True
        and replay_gate.get("passed") is True
        and replay_gate.get("status") == TORCH_TRAINING_REPLAY_MATCHED_STATUS
        and replay_gate.get("parity_status") == "matched"
        and backend.get("parity_status") == "matched"
    )


def _boundary_failures(boundary: dict[str, Any]) -> list[str]:
    return torch_training_attempt_boundary_failures(boundary)


def _check(name: str, passed: bool, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "reason": reason,
    }
