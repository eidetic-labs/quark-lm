"""Status resolution for optional PyTorch training parity attempts."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
)


TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS = "training_parity_matched"
TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS = (
    "blocked_pytorch_runtime"
)
TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_READY_STATUS = "ready_for_pytorch_parity"


def resolve_torch_training_parity_attempt_status(
    *,
    runtime: dict[str, Any],
    training_replay_parity_gate: dict[str, Any],
    training_parity_report: dict[str, Any],
) -> str:
    """Return the canonical attempt status for persisted evidence summaries."""

    _require_dict(runtime, "runtime")
    _require_dict(training_replay_parity_gate, "training_replay_parity_gate")
    _require_dict(training_parity_report, "training_parity_report")
    if not _runtime_preflight_ready(runtime):
        return str(
            runtime.get(
                "status",
                TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS,
            )
        )
    if not _replay_gate_matched(training_replay_parity_gate):
        return str(
            training_replay_parity_gate.get(
                "status",
                "training_parity_pending",
            )
        )
    if training_parity_report.get("passed") is True:
        return TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS
    return str(
        training_replay_parity_gate.get(
            "status",
            "training_parity_pending",
        )
    )


def resolve_torch_training_parity_attempt_passed(
    *,
    runtime: dict[str, Any],
    training_replay_parity_gate: dict[str, Any],
    training_parity_report: dict[str, Any],
) -> bool:
    """Return whether the attempt has satisfied all parity prerequisites."""

    _require_dict(runtime, "runtime")
    _require_dict(training_replay_parity_gate, "training_replay_parity_gate")
    _require_dict(training_parity_report, "training_parity_report")
    return (
        _runtime_preflight_ready(runtime)
        and _replay_gate_matched(training_replay_parity_gate)
        and training_parity_report.get("passed") is True
    )


def _require_dict(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")


def _runtime_preflight_ready(runtime: dict[str, Any]) -> bool:
    return (
        runtime.get("passed") is True
        and runtime.get("parity_attempt_allowed") is True
        and runtime.get("status") == TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_READY_STATUS
    )


def _replay_gate_matched(gate: dict[str, Any]) -> bool:
    return (
        gate.get("passed") is True
        and gate.get("status") == TORCH_TRAINING_REPLAY_MATCHED_STATUS
    )
