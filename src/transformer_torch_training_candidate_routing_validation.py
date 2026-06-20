"""Routing validation for PyTorch training parity candidates."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_candidate import (
    TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
)
from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS
from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
)


TORCH_TRAINING_CANDIDATE_ROUTE_FIELDS = (
    "implementation_status",
    "backend.parity_status",
    "training_case.status",
)


def validate_torch_training_candidate_routing(candidate: dict[str, Any]) -> None:
    """Validate top-level candidate status routing against runtime evidence."""

    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a dict")
    expected = _expected_routing(candidate)
    for field, expected_value in expected.items():
        actual = _route_value(candidate, field)
        if actual != expected_value:
            raise ValueError(f"candidate.{field} is inconsistent")


def _expected_routing(candidate: dict[str, Any]) -> dict[str, str]:
    runtime = _required_dict(candidate, "runtime")
    readiness = _required_dict(candidate, "training_readiness")
    gate = _required_dict(candidate, "training_replay_parity_gate")
    if runtime.get("available") is not True:
        return _routing("runtime_unavailable", "failed", "blocked")
    if runtime.get("dtype_available") is not True:
        return _routing("dtype_unavailable", "pending", "pending")
    if readiness.get("status") != TORCH_TRAINING_READY_STATUS:
        return _routing(TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS, "pending", "pending")
    if gate.get("status") == TORCH_TRAINING_REPLAY_MATCHED_STATUS:
        return _routing(TORCH_TRAINING_REPLAY_MATCHED_STATUS, "matched", "matched")
    return _routing(TORCH_TRAINING_REPLAY_PENDING_STATUS, "pending", "pending")


def _routing(
    implementation_status: str,
    parity_status: str,
    training_case_status: str,
) -> dict[str, str]:
    return {
        "implementation_status": implementation_status,
        "backend.parity_status": parity_status,
        "training_case.status": training_case_status,
    }


def _route_value(candidate: dict[str, Any], field: str) -> Any:
    if field == "implementation_status":
        return candidate.get("implementation_status")
    if field == "backend.parity_status":
        return _required_dict(candidate, "backend").get("parity_status")
    if field == "training_case.status":
        return _required_dict(candidate, "training_case").get("status")
    raise ValueError(f"candidate route field is unsupported: {field}")


def _required_dict(candidate: dict[str, Any], key: str) -> dict[str, Any]:
    value = candidate.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"candidate.{key} must be a dict")
    return value
