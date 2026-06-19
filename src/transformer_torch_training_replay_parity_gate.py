"""Aggregate replay parity gates for PyTorch training candidates."""

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
    TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
    TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS,
)
from transformer_torch_replay_final_evaluation import (
    TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
    TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
)
from transformer_torch_replay_update_comparison import (
    TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_UPDATE_MATCHED_STATUS,
)
from transformer_torch_runtime import TORCH_RUNTIME_KIND_PYTORCH
from transformer_torch_training_replay_gate_checks import (
    build_bool_check,
    build_replay_control_count_check,
    build_replay_probe_check,
    build_status_check,
    probe_status,
)
from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS


TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION = 1
TORCH_TRAINING_REPLAY_MATCHED_STATUS = "training_replay_parity_matched"
TORCH_TRAINING_REPLAY_PENDING_STATUS = "training_replay_parity_pending"
TORCH_TRAINING_REPLAY_BLOCKED_STATUS = "training_replay_parity_blocked"


def build_torch_training_replay_parity_gate(
    *,
    runtime: dict[str, Any],
    readiness: dict[str, Any],
    probes: dict[str, Any],
) -> dict[str, Any]:
    """Summarize whether replay evidence can count as training parity."""

    checks = [
        build_bool_check("runtime_available", runtime.get("available")),
        build_status_check(
            "runtime_kind",
            runtime.get("runtime_kind"),
            TORCH_RUNTIME_KIND_PYTORCH,
        ),
        build_bool_check("dtype_available", runtime.get("dtype_available")),
        build_status_check(
            "training_readiness",
            readiness.get("status"),
            TORCH_TRAINING_READY_STATUS,
        ),
        build_status_check(
            "initial_loss",
            probe_status(probes, "initial_loss_probe"),
            "matched",
        ),
        build_status_check(
            "backward",
            probe_status(probes, "backward_probe"),
            "gradients_available",
        ),
        build_status_check(
            "optimizer_step_readiness",
            probe_status(probes, "optimizer_step_probe"),
            "ready_for_optimizer_execution",
        ),
        build_status_check(
            "optimizer_step_control",
            probe_status(probes, "optimizer_step_execution_probe"),
            "step_control_matched",
        ),
        build_status_check(
            "replay_control",
            probe_status(probes, "accumulation_replay_control_probe"),
            "accumulation_replay_control_recorded",
        ),
        build_replay_control_count_check(
            "replay_gradient_signatures",
            probes.get("accumulation_replay_control_probe", {}),
            TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
        ),
        build_replay_probe_check(
            "replay_buffer",
            probes.get("accumulation_replay_buffer_comparison", {}),
            TORCH_REPLAY_BUFFER_MATCHED_STATUS,
            TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
            ["buffered_gradient_parity_proven"],
        ),
        build_replay_probe_check(
            "replay_update",
            probes.get("accumulation_replay_update_comparison", {}),
            TORCH_REPLAY_UPDATE_MATCHED_STATUS,
            TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
            ["optimizer_update_parity_proven"],
        ),
        build_replay_probe_check(
            "replay_final_evaluation",
            probes.get("accumulation_replay_final_evaluation", {}),
            TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
            TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
            ["final_logit_parity_proven", "final_loss_parity_proven"],
        ),
        build_replay_probe_check(
            "replay_checkpoint",
            probes.get("accumulation_replay_checkpoint_compatibility", {}),
            TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS,
            TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
            ["checkpoint_parity_proven"],
        ),
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION,
        "status": _status(runtime=runtime, passed=passed),
        "passed": passed,
        "parity_status": _parity_status(runtime=runtime, passed=passed),
        "implementation_status": _implementation_status(
            runtime=runtime,
            passed=passed,
        ),
        "promoted_training_backend": False,
        "checks": checks,
        "summary": _summary(checks),
        "reason": _reason(passed),
    }


def _status(*, runtime: dict[str, Any], passed: bool) -> str:
    if passed:
        return TORCH_TRAINING_REPLAY_MATCHED_STATUS
    if not runtime.get("available"):
        return TORCH_TRAINING_REPLAY_BLOCKED_STATUS
    return TORCH_TRAINING_REPLAY_PENDING_STATUS


def _parity_status(*, runtime: dict[str, Any], passed: bool) -> str:
    if passed:
        return "matched"
    if not runtime.get("available"):
        return "failed"
    return "pending"


def _implementation_status(*, runtime: dict[str, Any], passed: bool) -> str:
    if passed:
        return TORCH_TRAINING_REPLAY_MATCHED_STATUS
    if not runtime.get("available"):
        return "runtime_unavailable"
    return TORCH_TRAINING_REPLAY_PENDING_STATUS


def _summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [check["name"] for check in checks if check["passed"] is not True]
    return {
        "check_count": len(checks),
        "passed_check_count": len(checks) - len(failed),
        "failed_checks": failed,
    }


def _reason(passed: bool) -> str:
    return (
        "all replay parity gates match scalar training evidence"
        if passed
        else "one or more replay parity gates have not matched scalar training evidence"
    )
