"""Validation helpers for transformer backend parity fixtures."""

from __future__ import annotations

from typing import Any

from transformer_backend_parity_schema import (
    PARITY_FIXTURE_KIND,
    PARITY_SCHEMA_VERSION,
)
from transformer_backend_policy import (
    SCALAR_BACKEND,
    validate_transformer_backend_metadata,
)
from transformer_model import TRANSFORMER_ARCHITECTURE


def validate_backend_parity_fixture(fixture: dict[str, Any]) -> None:
    if fixture.get("schema_version") != PARITY_SCHEMA_VERSION:
        raise ValueError("unsupported backend parity fixture schema_version")
    if fixture.get("kind") != PARITY_FIXTURE_KIND:
        raise ValueError(f"kind must be {PARITY_FIXTURE_KIND}")
    if not fixture.get("fixture_id"):
        raise ValueError("fixture_id is required")
    if fixture.get("architecture") != TRANSFORMER_ARCHITECTURE:
        raise ValueError(f"architecture must be {TRANSFORMER_ARCHITECTURE}")
    if not isinstance(fixture.get("model_config"), dict):
        raise ValueError("model_config must be a dict")
    if not isinstance(fixture.get("weights"), dict):
        raise ValueError("weights must be a dict")
    validate_transformer_backend_metadata(
        fixture.get("reference_backend", {}),
        require_artifact_fields=True,
    )
    if fixture["reference_backend"].get("backend") != SCALAR_BACKEND:
        raise ValueError("reference_backend must be scalar_python")
    _validate_tolerance(fixture.get("tolerance"))
    _validate_forward_cases(fixture.get("forward_cases"))
    _validate_generation_cases(fixture.get("generation_cases"))


def _validate_tolerance(tolerance: Any) -> None:
    if not isinstance(tolerance, dict):
        raise ValueError("tolerance must be a dict")
    for key in ("absolute", "relative"):
        value = tolerance.get(key)
        if not isinstance(value, int | float) or value < 0.0:
            raise ValueError(f"tolerance.{key} must be a non-negative number")


def _validate_forward_cases(cases: Any) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError("forward_cases must be a non-empty list")
    for case in cases:
        for key in ("case_id", "context", "target", "logits", "loss"):
            if key not in case:
                raise ValueError(f"forward case missing {key}")


def _validate_generation_cases(cases: Any) -> None:
    if not isinstance(cases, list):
        raise ValueError("generation_cases must be a list")
    for case in cases:
        for key in ("case_id", "prompt", "text", "token_ids"):
            if key not in case:
                raise ValueError(f"generation case missing {key}")
