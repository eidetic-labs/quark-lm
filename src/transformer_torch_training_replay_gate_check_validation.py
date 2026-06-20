"""Per-check validation for PyTorch training replay parity gates."""

from __future__ import annotations

from typing import Any

from transformer_torch_accumulation_replay_control import (
    TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
)
from transformer_torch_replay_buffer_comparison import (
    TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_BUFFER_MATCHED_STATUS,
)
from transformer_torch_replay_checkpoint_compatibility import (
    TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS,
    TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
)
from transformer_torch_replay_final_evaluation import (
    TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
    TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
)
from transformer_torch_replay_update_comparison import (
    TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_UPDATE_MATCHED_STATUS,
)
from transformer_torch_runtime import TORCH_RUNTIME_KIND_PYTORCH
from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS


BOOLEAN_REPLAY_GATE_CHECKS = ("runtime_available", "dtype_available")
STATUS_REPLAY_GATE_CHECKS = {
    "runtime_kind": TORCH_RUNTIME_KIND_PYTORCH,
    "training_readiness": TORCH_TRAINING_READY_STATUS,
    "initial_loss": "matched",
    "backward": "gradients_available",
    "optimizer_step_readiness": "ready_for_optimizer_execution",
    "optimizer_step_control": "step_control_matched",
    "replay_control": "accumulation_replay_control_recorded",
}
REPLAY_CONTROL_COUNT_CHECK = "replay_gradient_signatures"
REPLAY_PROBE_GATE_CHECKS = {
    "replay_buffer": (
        TORCH_REPLAY_BUFFER_MATCHED_STATUS,
        TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
        ("buffered_gradient_parity_proven",),
    ),
    "replay_update": (
        TORCH_REPLAY_UPDATE_MATCHED_STATUS,
        TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
        ("optimizer_update_parity_proven",),
    ),
    "replay_final_evaluation": (
        TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
        TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
        ("final_logit_parity_proven", "final_loss_parity_proven"),
    ),
    "replay_checkpoint": (
        TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS,
        TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
        ("checkpoint_parity_proven",),
    ),
}


def validate_torch_training_replay_gate_check(check: dict[str, Any]) -> None:
    """Validate a persisted aggregate replay-gate check payload."""

    name = check["name"]
    if name in BOOLEAN_REPLAY_GATE_CHECKS:
        _validate_bool_check(check)
        return
    if name in STATUS_REPLAY_GATE_CHECKS:
        _validate_status_check(check, STATUS_REPLAY_GATE_CHECKS[name])
        return
    if name == REPLAY_CONTROL_COUNT_CHECK:
        _validate_replay_control_count_check(check)
        return
    if name in REPLAY_PROBE_GATE_CHECKS:
        expected_status, expected_schema, proof_flags = REPLAY_PROBE_GATE_CHECKS[
            name
        ]
        _validate_replay_probe_check(
            check,
            expected_status=expected_status,
            expected_schema_version=expected_schema,
            expected_proof_flags=proof_flags,
        )
        return
    raise ValueError(f"{_label(check)} is unsupported")


def _validate_bool_check(check: dict[str, Any]) -> None:
    actual = _require_bool(check, "actual")
    _require_passed_value(check, actual)


def _validate_status_check(check: dict[str, Any], expected: str) -> None:
    if check.get("expected") != expected:
        raise ValueError(f"{_label(check)}.expected is inconsistent")
    _require_passed_value(check, check.get("actual") == expected)


def _validate_replay_control_count_check(check: dict[str, Any]) -> None:
    if check.get("expected_schema_version") != (
        TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION
    ):
        raise ValueError(f"{_label(check)}.expected_schema_version is inconsistent")
    count_types_valid = _require_bool(check, "count_types_valid")
    planned = _require_nonnegative_int(check, "planned_count")
    executed = _require_nonnegative_int(check, "executed_count")
    backward = _require_nonnegative_int(check, "backward_count")
    matches = _require_nonnegative_int(check, "match_count")
    mismatches = _require_nonnegative_int(check, "mismatch_count")
    microsteps = _require_nonnegative_int(check, "microstep_count")
    passed = (
        check.get("schema_version") == check["expected_schema_version"]
        and count_types_valid
        and planned > 0
        and executed == planned
        and backward == planned
        and matches == planned
        and mismatches == 0
        and microsteps == planned
    )
    _require_passed_value(check, passed)


def _validate_replay_probe_check(
    check: dict[str, Any],
    *,
    expected_status: str,
    expected_schema_version: int,
    expected_proof_flags: tuple[str, ...],
) -> None:
    if check.get("expected") != expected_status:
        raise ValueError(f"{_label(check)}.expected is inconsistent")
    if check.get("expected_schema_version") != expected_schema_version:
        raise ValueError(f"{_label(check)}.expected_schema_version is inconsistent")
    proof_flags = check.get("proof_flags")
    if not isinstance(proof_flags, dict):
        raise ValueError(f"{_label(check)}.proof_flags must be a dict")
    if tuple(proof_flags) != expected_proof_flags:
        raise ValueError(f"{_label(check)}.proof_flags catalog is inconsistent")
    for flag, value in proof_flags.items():
        if not isinstance(value, bool):
            raise ValueError(f"{_label(check)}.proof_flags.{flag} is invalid")
    probe_passed = _require_bool(check, "probe_passed")
    passed = (
        probe_passed
        and check.get("status") == expected_status
        and check.get("schema_version") == expected_schema_version
        and all(proof_flags.values())
    )
    _require_passed_value(check, passed)


def _require_passed_value(check: dict[str, Any], expected: bool) -> None:
    if check.get("passed") is not expected:
        raise ValueError(f"{_label(check)}.passed is inconsistent")


def _require_bool(check: dict[str, Any], key: str) -> bool:
    value = check.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{_label(check)}.{key} must be a bool")
    return value


def _require_nonnegative_int(check: dict[str, Any], key: str) -> int:
    value = check.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{_label(check)}.{key} must be a nonnegative int")
    return value


def _label(check: dict[str, Any]) -> str:
    return f"candidate.training_replay_parity_gate.checks.{check['name']}"
