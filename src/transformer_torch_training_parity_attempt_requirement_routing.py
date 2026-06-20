"""Shared routing validation for PyTorch parity attempt requirements."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_requirements import (
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
)


def validate_torch_training_requirement_routing(
    *,
    stage: str,
    status: str,
    next_actions: list[str],
    runtime_status: Any,
    primary_blockers: list[str] | None = None,
    exact_actions: bool = False,
    require_blockers: bool = False,
    stage_error: str = "next_requirements.stage",
    status_error: str = "next_requirements.status",
    action_error: str = "next_requirements.next_actions",
    blocker_error: str = "next_requirements.primary_blockers",
) -> None:
    """Validate requirement stage, status, and remediation-action routing."""

    if stage not in TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES:
        raise ValueError(f"{stage_error} is unsupported")
    if status not in allowed_torch_training_requirement_statuses(stage):
        raise ValueError(f"{status_error} is unsupported")
    if stage == "complete":
        if next_actions:
            raise ValueError(f"{action_error} are inconsistent")
        return
    if require_blockers and not primary_blockers:
        raise ValueError(f"{blocker_error} must not be empty")
    if not next_actions:
        raise ValueError(f"{action_error} must not be empty")
    if stage == "runtime_preflight":
        _validate_runtime_preflight_routing(
            runtime_status=runtime_status,
            next_actions=next_actions,
            primary_blockers=primary_blockers,
            action_error=action_error,
            blocker_error=blocker_error,
            require_blockers=require_blockers,
        )
        return
    prefix = torch_training_requirement_action_prefix(stage, stage_error=stage_error)
    if exact_actions and primary_blockers is not None:
        expected_actions = [f"{prefix}:{blocker}" for blocker in primary_blockers]
        if next_actions != expected_actions:
            raise ValueError(f"{action_error} do not match stage blockers")
        return
    if any(not _has_action_prefix(action, prefix) for action in next_actions):
        raise ValueError(f"{action_error} are inconsistent")


def allowed_torch_training_requirement_statuses(stage: str) -> tuple[str, ...]:
    """Return allowed status values for a next-requirements stage."""

    if stage == "complete":
        return ("satisfied",)
    if stage == "runtime_preflight":
        return ("blocked",)
    if stage == "training_readiness":
        return ("blocked", "pending")
    return ("pending",)


def torch_training_requirement_action_prefix(
    stage: str,
    *,
    stage_error: str = "next_requirements.stage",
) -> str:
    """Return the action prefix associated with a next-requirements stage."""

    if stage == "training_readiness":
        return "satisfy_training_readiness"
    if stage == "training_replay_parity":
        return "resolve_replay_gate"
    if stage == "training_parity_report":
        return "resolve_training_parity_check"
    raise ValueError(f"{stage_error} is unsupported")


def _validate_runtime_preflight_routing(
    *,
    runtime_status: Any,
    next_actions: list[str],
    primary_blockers: list[str] | None,
    action_error: str,
    blocker_error: str,
    require_blockers: bool,
) -> None:
    if len(next_actions) != 1:
        raise ValueError(f"{action_error} runtime action count is invalid")
    if next_actions[0] not in TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS:
        raise ValueError(f"{action_error} runtime action is unsupported")
    expected_action = TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS.get(
        runtime_status,
        "fix_pytorch_runtime_preflight",
    )
    if next_actions[0] != expected_action:
        raise ValueError(f"{action_error} runtime action is inconsistent")
    expected_blocker = TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS.get(
        runtime_status,
    )
    if (
        require_blockers
        and expected_blocker is not None
        and primary_blockers is not None
        and expected_blocker not in primary_blockers
    ):
        raise ValueError(f"{blocker_error} runtime blocker is inconsistent")


def _has_action_prefix(action: str, prefix: str) -> bool:
    marker = f"{prefix}:"
    return action.startswith(marker) and len(action) > len(marker)
