"""Standalone validation for PyTorch training parity candidates."""

from __future__ import annotations

from typing import Any

from transformer_backend_policy import (
    PYTORCH_BACKEND,
    validate_transformer_backend_metadata,
)
from transformer_torch_runtime_report_check import build_torch_runtime_report_check
from transformer_torch_training_case_validation import validate_torch_training_case
from transformer_torch_training_candidate import (
    TORCH_TRAINING_PARITY_CANDIDATE_KIND,
    TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION,
    TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
)
from transformer_torch_training_readiness import (
    TORCH_TRAINING_READY_STATUS,
)
from transformer_torch_training_readiness_validation import (
    validate_torch_training_readiness,
)
from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
    TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION,
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
)


REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS = (
    "schema_version",
    "kind",
    "fixture_id",
    "implementation_status",
    "backend",
    "runtime",
    "runtime_report",
    "model_config",
    "tokenizer",
    "optimizer_config",
    "parameter_manifest",
    "optimizer_step_contract",
    "training_readiness",
    "training_state",
    "initial_loss_probe",
    "backward_probe",
    "accumulation_replay_plan",
    "accumulation_replay_control_probe",
    "accumulation_replay_buffer_comparison",
    "accumulation_replay_update_comparison",
    "accumulation_replay_final_evaluation",
    "accumulation_replay_checkpoint_compatibility",
    "optimizer_step_probe",
    "optimizer_step_execution_probe",
    "training_replay_parity_gate",
    "training_case",
)


def validate_torch_training_parity_candidate(candidate: dict[str, Any]) -> None:
    """Validate the top-level PyTorch training candidate evidence contract."""

    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a dict")
    _require_keys(candidate, REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS)
    if (
        candidate.get("schema_version")
        != TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION
    ):
        raise ValueError("candidate.schema_version is inconsistent")
    if candidate.get("kind") != TORCH_TRAINING_PARITY_CANDIDATE_KIND:
        raise ValueError("candidate.kind is inconsistent")
    _require_non_empty_string(candidate, "fixture_id")
    _require_non_empty_string(candidate, "implementation_status")
    _require_dicts(candidate, _DICT_SECTIONS)
    _validate_backend(candidate["backend"])
    _validate_runtime_report(candidate)
    validate_torch_training_readiness(candidate["training_readiness"])
    _validate_replay_gate(candidate["training_replay_parity_gate"])
    validate_torch_training_case(candidate["training_case"])
    _validate_routing(candidate)


def _validate_backend(backend: dict[str, Any]) -> None:
    validate_transformer_backend_metadata(backend, require_artifact_fields=True)
    if backend.get("backend") != PYTORCH_BACKEND:
        raise ValueError("candidate.backend must be pytorch")


def _validate_runtime_report(candidate: dict[str, Any]) -> None:
    check = build_torch_runtime_report_check(
        runtime_report=candidate["runtime_report"],
        runtime=candidate["runtime"],
    )
    if check.get("passed") is True:
        return
    failures = check.get("failed_runtime_checks") or [check.get("error", "invalid")]
    raise ValueError(f"candidate.runtime_report.{failures[0]} is inconsistent")


def _validate_replay_gate(gate: dict[str, Any]) -> None:
    if gate.get("schema_version") != TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION:
        raise ValueError("candidate.training_replay_parity_gate.schema_version")
    if gate.get("status") not in {
        TORCH_TRAINING_REPLAY_MATCHED_STATUS,
        TORCH_TRAINING_REPLAY_PENDING_STATUS,
        TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
    }:
        raise ValueError("candidate.training_replay_parity_gate.status is invalid")
    if not isinstance(gate.get("passed"), bool):
        raise ValueError("candidate.training_replay_parity_gate.passed is invalid")
    if gate.get("parity_status") not in {"matched", "pending", "failed"}:
        raise ValueError("candidate.training_replay_parity_gate.parity_status")
    _require_non_empty_string(gate, "implementation_status")
    if gate.get("promoted_training_backend") is not False:
        raise ValueError("candidate.training_replay_parity_gate must not promote")
    if not isinstance(gate.get("checks"), list):
        raise ValueError("candidate.training_replay_parity_gate.checks is invalid")
    _validate_summary("candidate.training_replay_parity_gate", gate.get("summary"))
    _require_non_empty_string(gate, "reason")
    _validate_replay_gate_status(gate)


def _validate_replay_gate_status(gate: dict[str, Any]) -> None:
    if gate["passed"] is True:
        if gate["status"] != TORCH_TRAINING_REPLAY_MATCHED_STATUS:
            raise ValueError("candidate.training_replay_parity_gate.status")
        if gate["parity_status"] != "matched":
            raise ValueError("candidate.training_replay_parity_gate.parity_status")
        if gate["summary"].get("failed_checks") != []:
            raise ValueError("candidate.training_replay_parity_gate.failed_checks")
    if gate["status"] == TORCH_TRAINING_REPLAY_BLOCKED_STATUS:
        if gate["parity_status"] != "failed":
            raise ValueError("candidate.training_replay_parity_gate.parity_status")
    if gate["status"] == TORCH_TRAINING_REPLAY_PENDING_STATUS:
        if gate["parity_status"] != "pending":
            raise ValueError("candidate.training_replay_parity_gate.parity_status")


def _validate_routing(candidate: dict[str, Any]) -> None:
    expected = _expected_routing(candidate)
    for key, value in expected.items():
        if key == "backend.parity_status":
            actual = candidate["backend"].get("parity_status")
        elif key == "training_case.status":
            actual = candidate["training_case"].get("status")
        else:
            actual = candidate.get(key)
        if actual != value:
            raise ValueError(f"candidate.{key} is inconsistent")


def _expected_routing(candidate: dict[str, Any]) -> dict[str, str]:
    runtime = candidate["runtime"]
    readiness = candidate["training_readiness"]
    gate = candidate["training_replay_parity_gate"]
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


def _validate_summary(label: str, summary: Any) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"{label}.summary is invalid")
    if not isinstance(summary.get("failed_checks"), list):
        raise ValueError(f"{label}.summary.failed_checks is invalid")
    if any(not isinstance(item, str) for item in summary["failed_checks"]):
        raise ValueError(f"{label}.summary.failed_checks is invalid")


def _require_keys(record: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in record:
            raise ValueError(f"candidate.{key} is missing")


def _require_dicts(record: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if not isinstance(record.get(key), dict):
            raise ValueError(f"candidate.{key} must be a dict")


def _require_non_empty_string(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate.{key} must be a non-empty string")


_NON_DICT_CANDIDATE_KEYS = {
    "schema_version",
    "kind",
    "fixture_id",
    "implementation_status",
}
_DICT_SECTIONS = tuple(
    key for key in REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS
    if key not in _NON_DICT_CANDIDATE_KEYS
)
