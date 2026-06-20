"""Status resolution for optional PyTorch training parity attempts."""

from __future__ import annotations

from typing import Any


TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS = "training_parity_matched"
TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS = (
    "blocked_pytorch_runtime"
)


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
    if runtime.get("parity_attempt_allowed") is not True:
        return str(
            runtime.get(
                "status",
                TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS,
            )
        )
    if training_replay_parity_gate.get("passed") is not True:
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
        runtime.get("parity_attempt_allowed") is True
        and training_replay_parity_gate.get("passed") is True
        and training_parity_report.get("passed") is True
    )


def _require_dict(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
