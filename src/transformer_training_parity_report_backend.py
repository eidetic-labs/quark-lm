"""Backend checks for transformer training parity reports."""

from __future__ import annotations

from typing import Any

from transformer_backend_policy import (
    PYTORCH_BACKEND,
    validate_transformer_backend_metadata,
)
from transformer_torch_training_candidate import TORCH_TRAINING_PARITY_CANDIDATE_KIND


def build_backend_metadata_check(backend: Any) -> dict[str, Any]:
    """Validate backend metadata before a parity report trusts it."""

    if not isinstance(backend, dict):
        return {
            "name": "backend_metadata",
            "passed": False,
            "error": "backend metadata is missing",
        }
    try:
        validate_transformer_backend_metadata(
            backend,
            require_artifact_fields=backend.get("backend") == PYTORCH_BACKEND,
        )
    except ValueError as exc:
        return {
            "name": "backend_metadata",
            "passed": False,
            "backend": backend.get("backend"),
            "error": str(exc),
        }
    return {
        "name": "backend_metadata",
        "passed": True,
        "backend": backend.get("backend"),
        "parity_status": backend.get("parity_status"),
    }


def build_torch_candidate_backend_check(backend: Any) -> dict[str, Any]:
    """Ensure a PyTorch candidate kind cannot hide behind scalar metadata."""

    actual = backend.get("backend") if isinstance(backend, dict) else None
    return {
        "name": "pytorch_candidate_backend",
        "passed": actual == PYTORCH_BACKEND,
        "expected": PYTORCH_BACKEND,
        "actual": actual,
    }


def candidate_backend_name(candidate: dict[str, Any]) -> str | None:
    backend = candidate.get("backend")
    return backend.get("backend") if isinstance(backend, dict) else None


def candidate_kind(candidate: dict[str, Any]) -> str | None:
    kind = candidate.get("kind")
    return kind if isinstance(kind, str) else None


def is_torch_training_candidate(candidate: dict[str, Any]) -> bool:
    return candidate_kind(candidate) == TORCH_TRAINING_PARITY_CANDIDATE_KIND


def requires_pytorch_checks(candidate: dict[str, Any]) -> bool:
    return (
        candidate_backend_name(candidate) == PYTORCH_BACKEND
        or is_torch_training_candidate(candidate)
    )
