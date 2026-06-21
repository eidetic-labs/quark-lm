"""Validation for PyTorch training candidate case evidence."""

from __future__ import annotations

import math
from typing import Any


def validate_torch_training_case(case: dict[str, Any]) -> None:
    """Validate the candidate training-case routing payload."""

    if not isinstance(case, dict):
        raise ValueError("candidate.training_case must be a dict")
    _validate_case_keys(case)
    _require_non_empty_string(case, "case_id")
    _require_non_empty_string(case, "status")
    _require_non_empty_string(case, "reason")
    _validate_context(case)
    _validate_token_id(case, "target")
    _validate_learning_rate(case)
    _validate_steps(case)
    if case["status"] == "matched":
        _validate_matched_case(case)


def _validate_case_keys(case: dict[str, Any]) -> None:
    if case.get("status") == "matched":
        required = _MATCHED_CASE_KEYS
        allowed = _MATCHED_CASE_KEYS | _MATCHED_EVIDENCE_KEYS
    else:
        required = _BASE_CASE_KEYS
        allowed = _BASE_CASE_KEYS
    keys = set(case)
    missing = required - keys
    if missing:
        field = sorted(missing)[0]
        raise ValueError(f"candidate.training_case.{field} is missing")
    if keys - allowed:
        raise ValueError("candidate.training_case keys are inconsistent")


def _validate_context(case: dict[str, Any]) -> None:
    context = case.get("context")
    if not isinstance(context, list) or not context:
        raise ValueError("candidate.training_case.context is invalid")
    if any(not _is_non_negative_int(token_id) for token_id in context):
        raise ValueError("candidate.training_case.context is invalid")


def _validate_token_id(case: dict[str, Any], key: str) -> None:
    if not _is_non_negative_int(case.get(key)):
        raise ValueError(f"candidate.training_case.{key} is invalid")


def _validate_learning_rate(case: dict[str, Any]) -> None:
    value = case.get("learning_rate")
    if type(value) not in {int, float}:
        raise ValueError("candidate.training_case.learning_rate is invalid")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError("candidate.training_case.learning_rate is invalid")


def _validate_steps(case: dict[str, Any]) -> None:
    if not _is_positive_int(case.get("steps")):
        raise ValueError("candidate.training_case.steps is invalid")


def _validate_matched_case(case: dict[str, Any]) -> None:
    if case.get("promoted_training_backend") is not False:
        raise ValueError("candidate.training_case must not promote")
    if case.get("evidence_source") != "training_replay_parity_gate":
        raise ValueError("candidate.training_case.evidence_source is invalid")


def _is_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _require_non_empty_string(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate.training_case.{key} must be a non-empty string")


_BASE_CASE_KEYS = {
    "case_id",
    "status",
    "reason",
    "context",
    "target",
    "learning_rate",
    "steps",
}
_MATCHED_CASE_KEYS = _BASE_CASE_KEYS | {
    "evidence_source",
    "promoted_training_backend",
}
# A matched case also carries the scalar training evidence that the parity
# report compares against (transformer_training_parity_report.py): initial/final
# logits and loss, per-step records, optimizer state, and parameter signatures.
# These are optional known fields — present in the real replay-matched builder,
# absent in a minimal matched stub — and are distinct from unknown drift keys,
# which remain rejected.
_MATCHED_EVIDENCE_KEYS = {
    "initial_logits",
    "initial_loss",
    "final_logits",
    "final_loss",
    "step_records",
    "optimizer_state",
    "parameter_signature",
    "trainable_parameter_signature",
}
