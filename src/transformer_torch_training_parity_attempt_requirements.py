"""Requirement classification for PyTorch training parity attempts."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS


TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND = (
    "transformer_torch_training_parity_attempt_requirements"
)
TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION = 1
TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES = (
    "runtime_preflight",
    "training_readiness",
    "training_replay_parity",
    "training_parity_report",
    "complete",
)


def validate_torch_training_parity_attempt_requirements(
    requirements: dict[str, Any],
) -> None:
    """Validate the standalone next-requirements artifact shape."""

    if not isinstance(requirements, dict):
        raise ValueError("next_requirements must be a dict")
    if (
        requirements.get("schema_version")
        != TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION
    ):
        raise ValueError("next_requirements.schema_version is inconsistent")
    if requirements.get("kind") != TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND:
        raise ValueError("next_requirements.kind is inconsistent")
    stage = _required_string(requirements, "stage")
    status = _required_string(requirements, "status")
    if stage not in TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES:
        raise ValueError("next_requirements.stage is unsupported")
    if status not in _allowed_statuses(stage):
        raise ValueError("next_requirements.status is unsupported")
    _require_string_list(requirements, "primary_blockers")
    _require_string_list(requirements, "next_actions")
    if stage == "complete" and (
        requirements["primary_blockers"] or requirements["next_actions"]
    ):
        raise ValueError("next_requirements.complete actions must be empty")
    _require_keys(
        requirements,
        (
            "runtime_status",
            "parity_attempt_allowed",
            "training_readiness_status",
            "training_replay_parity_status",
            "training_report_passed",
        ),
    )
    if not isinstance(requirements["parity_attempt_allowed"], bool):
        raise ValueError("next_requirements.parity_attempt_allowed must be a bool")
    if not isinstance(requirements["training_report_passed"], bool):
        raise ValueError("next_requirements.training_report_passed must be a bool")


def build_torch_training_parity_attempt_requirements(
    *,
    runtime_report: dict[str, Any],
    candidate: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the first unsatisfied requirement for an attempt."""

    if report.get("passed") is True:
        return _requirements(
            stage="complete",
            status="satisfied",
            primary_blockers=[],
            next_actions=[],
            runtime_report=runtime_report,
            candidate=candidate,
            report=report,
        )
    if runtime_report.get("parity_attempt_allowed") is not True:
        return _requirements(
            stage="runtime_preflight",
            status="blocked",
            primary_blockers=_runtime_blockers(runtime_report),
            next_actions=_runtime_next_actions(runtime_report),
            runtime_report=runtime_report,
            candidate=candidate,
            report=report,
        )
    readiness = candidate.get("training_readiness", {})
    if readiness.get("status") != TORCH_TRAINING_READY_STATUS:
        blockers = _summary_failures(readiness)
        return _requirements(
            stage="training_readiness",
            status=_readiness_status(readiness),
            primary_blockers=blockers,
            next_actions=_prefixed("satisfy_training_readiness", blockers),
            runtime_report=runtime_report,
            candidate=candidate,
            report=report,
        )
    gate = candidate.get("training_replay_parity_gate", {})
    if gate.get("passed") is not True:
        blockers = _summary_failures(gate)
        return _requirements(
            stage="training_replay_parity",
            status="pending",
            primary_blockers=blockers,
            next_actions=_prefixed("resolve_replay_gate", blockers),
            runtime_report=runtime_report,
            candidate=candidate,
            report=report,
        )
    blockers = _summary_failures(report)
    return _requirements(
        stage="training_parity_report",
        status="pending",
        primary_blockers=blockers,
        next_actions=_prefixed("resolve_training_parity_check", blockers),
        runtime_report=runtime_report,
        candidate=candidate,
        report=report,
    )


def _requirements(
    *,
    stage: str,
    status: str,
    primary_blockers: list[str],
    next_actions: list[str],
    runtime_report: dict[str, Any],
    candidate: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    gate = candidate.get("training_replay_parity_gate", {})
    readiness = candidate.get("training_readiness", {})
    return {
        "schema_version": TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
        "kind": TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
        "stage": stage,
        "status": status,
        "primary_blockers": primary_blockers,
        "next_actions": next_actions,
        "runtime_status": runtime_report.get("status"),
        "parity_attempt_allowed": runtime_report.get("parity_attempt_allowed"),
        "training_readiness_status": readiness.get("status"),
        "training_replay_parity_status": gate.get("status"),
        "training_report_passed": report.get("passed"),
    }


def _runtime_blockers(runtime_report: dict[str, Any]) -> list[str]:
    blockers = _summary_failures(runtime_report)
    return blockers if blockers else ["parity_attempt_allowed"]


def _runtime_next_actions(runtime_report: dict[str, Any]) -> list[str]:
    status = runtime_report.get("status")
    if status == "blocked_runtime_unavailable":
        return ["install_real_pytorch_runtime"]
    if status == "blocked_test_double_runtime":
        return ["run_again_with_real_pytorch_runtime"]
    if status == "blocked_dtype_unavailable":
        return ["request_available_pytorch_dtype"]
    return ["fix_pytorch_runtime_preflight"]


def _readiness_status(readiness: dict[str, Any]) -> str:
    return "blocked" if readiness.get("status") == "blocked" else "pending"


def _summary_failures(payload: dict[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return []
    return list(summary.get("failed_checks", []))


def _prefixed(prefix: str, values: list[str]) -> list[str]:
    return [f"{prefix}:{value}" for value in values]


def _allowed_statuses(stage: str) -> tuple[str, ...]:
    if stage == "complete":
        return ("satisfied",)
    if stage == "runtime_preflight":
        return ("blocked",)
    if stage == "training_readiness":
        return ("blocked", "pending")
    return ("pending",)


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"next_requirements.{key} must be a non-empty string")
    return value


def _require_string_list(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"next_requirements.{key} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"next_requirements.{key} must contain strings")


def _require_keys(record: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in record:
            raise ValueError(f"next_requirements.{key} is missing")
