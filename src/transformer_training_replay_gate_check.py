"""Training replay-gate checks for PyTorch parity reports."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
)
from transformer_torch_training_replay_gate_validation import (
    validate_torch_training_replay_parity_gate,
)


def build_training_replay_parity_gate_check(gate: Any) -> dict[str, Any]:
    """Check that a training candidate has matched replay parity evidence."""

    if not isinstance(gate, dict):
        return {
            "name": "training_replay_parity_gate",
            "passed": False,
            "error": "training replay parity gate is missing",
        }
    try:
        validate_torch_training_replay_parity_gate(gate)
    except ValueError as exc:
        return {
            "name": "training_replay_parity_gate",
            "passed": False,
            "error": str(exc),
        }
    summary = gate.get("summary")
    failed_checks = (
        list(summary.get("failed_checks", [])) if isinstance(summary, dict) else []
    )
    passed = (
        gate.get("passed") is True
        and gate.get("status") == TORCH_TRAINING_REPLAY_MATCHED_STATUS
        and gate.get("parity_status") == "matched"
        and gate.get("promoted_training_backend") is False
    )
    return {
        "name": "training_replay_parity_gate",
        "passed": passed,
        "status": gate.get("status"),
        "parity_status": gate.get("parity_status"),
        "promoted_training_backend": gate.get("promoted_training_backend"),
        "failed_gate_checks": failed_checks,
    }
