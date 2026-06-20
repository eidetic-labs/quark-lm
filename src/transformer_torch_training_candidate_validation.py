"""Standalone validation for PyTorch training parity candidates."""

from __future__ import annotations

from typing import Any

from transformer_backend_policy import (
    PYTORCH_BACKEND,
    validate_transformer_backend_metadata,
)
from transformer_torch_training_case_validation import validate_torch_training_case
from transformer_torch_training_candidate import (
    TORCH_TRAINING_PARITY_CANDIDATE_KIND,
    TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION,
)
from transformer_torch_training_candidate_runtime_validation import (
    validate_torch_training_candidate_runtime_report,
)
from transformer_torch_training_candidate_routing_validation import (
    validate_torch_training_candidate_routing,
)
from transformer_torch_training_readiness_validation import (
    validate_torch_training_readiness,
)
from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
)
from transformer_torch_training_replay_gate_validation import (
    validate_torch_training_replay_parity_gate,
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
    validate_torch_training_candidate_runtime_report(candidate)
    validate_torch_training_readiness(candidate["training_readiness"])
    validate_torch_training_replay_parity_gate(
        candidate["training_replay_parity_gate"]
    )
    validate_torch_training_case(candidate["training_case"])
    validate_torch_training_candidate_routing(candidate)


def _validate_backend(backend: dict[str, Any]) -> None:
    validate_transformer_backend_metadata(backend, require_artifact_fields=True)
    if backend.get("backend") != PYTORCH_BACKEND:
        raise ValueError("candidate.backend must be pytorch")


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
