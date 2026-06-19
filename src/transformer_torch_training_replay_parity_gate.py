"""Aggregate replay parity gates for PyTorch training candidates."""

from __future__ import annotations

from typing import Any

from transformer_torch_replay_buffer_comparison import (
    TORCH_REPLAY_BUFFER_MATCHED_STATUS,
)
from transformer_torch_replay_checkpoint_compatibility import (
    TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS,
)
from transformer_torch_replay_final_evaluation import (
    TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
)
from transformer_torch_replay_update_comparison import (
    TORCH_REPLAY_UPDATE_MATCHED_STATUS,
)
from transformer_torch_runtime import TORCH_RUNTIME_KIND_PYTORCH
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
        _bool_check("runtime_available", runtime.get("available")),
        _status_check(
            "runtime_kind",
            runtime.get("runtime_kind"),
            TORCH_RUNTIME_KIND_PYTORCH,
        ),
        _bool_check("dtype_available", runtime.get("dtype_available")),
        _status_check(
            "training_readiness",
            readiness.get("status"),
            TORCH_TRAINING_READY_STATUS,
        ),
        _status_check(
            "initial_loss",
            _probe_status(probes, "initial_loss_probe"),
            "matched",
        ),
        _status_check(
            "backward",
            _probe_status(probes, "backward_probe"),
            "gradients_available",
        ),
        _status_check(
            "optimizer_step_readiness",
            _probe_status(probes, "optimizer_step_probe"),
            "ready_for_optimizer_execution",
        ),
        _status_check(
            "optimizer_step_control",
            _probe_status(probes, "optimizer_step_execution_probe"),
            "step_control_matched",
        ),
        _status_check(
            "replay_control",
            _probe_status(probes, "accumulation_replay_control_probe"),
            "accumulation_replay_control_recorded",
        ),
        _count_check(
            "replay_gradient_signatures",
            probes.get("accumulation_replay_control_probe", {}),
        ),
        _passed_probe_check(
            "replay_buffer",
            probes.get("accumulation_replay_buffer_comparison", {}),
            TORCH_REPLAY_BUFFER_MATCHED_STATUS,
        ),
        _passed_probe_check(
            "replay_update",
            probes.get("accumulation_replay_update_comparison", {}),
            TORCH_REPLAY_UPDATE_MATCHED_STATUS,
        ),
        _passed_probe_check(
            "replay_final_evaluation",
            probes.get("accumulation_replay_final_evaluation", {}),
            TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
        ),
        _passed_probe_check(
            "replay_checkpoint",
            probes.get("accumulation_replay_checkpoint_compatibility", {}),
            TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS,
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


def _probe_status(probes: dict[str, Any], name: str) -> Any:
    probe = probes.get(name, {})
    return probe.get("status") if isinstance(probe, dict) else None


def _bool_check(name: str, value: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(value), "actual": bool(value)}


def _status_check(name: str, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _count_check(name: str, probe: dict[str, Any]) -> dict[str, Any]:
    match_count = int(probe.get("gradient_signature_match_count", 0))
    mismatch_count = int(probe.get("gradient_signature_mismatch_count", 0))
    planned_count = int(probe.get("planned_microstep_count", 0))
    passed = (
        planned_count > 0
        and match_count == planned_count
        and mismatch_count == 0
    )
    return {
        "name": name,
        "passed": passed,
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "planned_count": planned_count,
    }


def _passed_probe_check(
    name: str,
    probe: dict[str, Any],
    expected_status: str,
) -> dict[str, Any]:
    actual_status = probe.get("status")
    return {
        "name": name,
        "passed": bool(probe.get("passed")) and actual_status == expected_status,
        "expected": expected_status,
        "status": actual_status,
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
