"""Closed-world runtime backend policy for transformer experiments."""

from __future__ import annotations

from typing import Any


SCALAR_BACKEND = "scalar_python"
PYTORCH_BACKEND = "pytorch"
PLANNED_PERFORMANCE_BACKEND = PYTORCH_BACKEND
ALLOWED_BACKENDS = {SCALAR_BACKEND, PYTORCH_BACKEND}
SCALAR_PARITY_STATUS = "reference"
PYTORCH_PARITY_STATUSES = {"pending", "matched", "failed"}


def transformer_backend_metadata(
    *,
    active_backend: str = SCALAR_BACKEND,
    seed: int | None = None,
    tokenizer_type: str | None = None,
    corpus_hash: str | None = None,
    tokenizer_manifest_hash: str | None = None,
    device: str = "cpu",
    dtype: str = "float64",
    parity_status: str | None = None,
) -> dict[str, Any]:
    """Build the policy fields every transformer backend artifact must expose."""

    backend = _normalize_backend(active_backend)
    status = parity_status or (
        SCALAR_PARITY_STATUS if backend == SCALAR_BACKEND else "pending"
    )
    return {
        "backend": backend,
        "backend_role": _backend_role(backend),
        "planned_performance_backend": PLANNED_PERFORMANCE_BACKEND,
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "tokenizer_type": tokenizer_type,
        "tokenizer_manifest_hash": tokenizer_manifest_hash,
        "corpus_hash": corpus_hash,
        "requires_scalar_parity": backend != SCALAR_BACKEND,
        "parity_status": status,
        "runtime_library_allowed": True,
        "purity": {
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "copied_model_code": False,
            "unledgered_training_data": False,
        },
    }


def validate_transformer_backend_metadata(
    metadata: dict[str, Any],
    *,
    require_artifact_fields: bool = False,
) -> None:
    """Validate backend metadata before treating an artifact as evidence."""

    backend = metadata.get("backend")
    if backend not in ALLOWED_BACKENDS:
        raise ValueError("backend must be scalar_python or pytorch")
    if metadata.get("planned_performance_backend") != PLANNED_PERFORMANCE_BACKEND:
        raise ValueError("planned_performance_backend must be pytorch")
    if not isinstance(metadata.get("runtime_library_allowed"), bool):
        raise ValueError("runtime_library_allowed must be a bool")
    if metadata.get("requires_scalar_parity") is not (backend != SCALAR_BACKEND):
        raise ValueError("requires_scalar_parity does not match backend")
    _validate_purity(metadata.get("purity"))
    _validate_parity_status(backend, metadata.get("parity_status"))
    if require_artifact_fields:
        _validate_required_artifact_fields(metadata)


def _normalize_backend(active_backend: str) -> str:
    backend = active_backend.strip().lower().replace("-", "_")
    if backend not in ALLOWED_BACKENDS:
        raise ValueError("backend must be scalar_python or pytorch")
    return backend


def _backend_role(backend: str) -> str:
    if backend == SCALAR_BACKEND:
        return "canonical_reference"
    return "experimental_performance"


def _validate_purity(purity: Any) -> None:
    if not isinstance(purity, dict):
        raise ValueError("purity metadata must be a dict")
    for key in (
        "pretrained_weights",
        "pretrained_tokenizer",
        "external_embeddings",
        "copied_model_code",
        "unledgered_training_data",
    ):
        if purity.get(key) is not False:
            raise ValueError(f"purity.{key} must be false")


def _validate_parity_status(backend: str, status: Any) -> None:
    if backend == SCALAR_BACKEND and status != SCALAR_PARITY_STATUS:
        raise ValueError("scalar_python backend parity_status must be reference")
    if backend == PYTORCH_BACKEND and status not in PYTORCH_PARITY_STATUSES:
        raise ValueError("pytorch backend parity_status must be pending, matched, or failed")


def _validate_required_artifact_fields(metadata: dict[str, Any]) -> None:
    for key in ("seed", "tokenizer_type", "corpus_hash"):
        if metadata.get(key) in (None, ""):
            raise ValueError(f"{key} is required for backend artifacts")
    if metadata.get("backend") == PYTORCH_BACKEND:
        for key in ("device", "dtype", "parity_status"):
            if metadata.get(key) in (None, ""):
                raise ValueError(f"{key} is required for pytorch backend artifacts")
