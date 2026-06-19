"""Promotion boundary for optional PyTorch training parity evidence."""

from __future__ import annotations

from typing import Any


TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION = 1
TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS = "training_backend_not_promoted"


def build_torch_training_backend_promotion_gate(
    *,
    candidate: dict[str, Any],
    report: dict[str, Any],
    closed_world_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Record why replay parity evidence is not backend promotion evidence."""

    checks = [
        _check(
            "training_parity_report",
            report.get("passed") is True,
            "training parity report must pass before promotion can be considered",
        ),
        _check(
            "closed_world_boundary",
            _boundary_passed(closed_world_boundary),
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
        "closed_world_boundary_passed": _boundary_passed(closed_world_boundary),
        "checks": checks,
        "blockers": [check["name"] for check in checks if not check["passed"]],
        "required_future_gates": [
            "general_training_backend_gate",
            "model_quality_eval_gate",
            "profile_and_retention_gate",
        ],
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
        and replay_gate.get("parity_status") == "matched"
        and backend.get("parity_status") == "matched"
    )


def _boundary_passed(boundary: dict[str, Any]) -> bool:
    forbidden = (
        "learned_assets_imported",
        "training_data_imported",
        "pretrained_weights_imported",
        "pretrained_tokenizer_imported",
        "external_embeddings_imported",
    )
    return all(boundary.get(key) is False for key in forbidden)


def _check(name: str, passed: bool, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "reason": reason,
    }
