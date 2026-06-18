"""PyTorch training parity candidate artifacts for transformer experiments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from transformer_backend_policy import PYTORCH_BACKEND, transformer_backend_metadata
from transformer_torch_runtime import TorchImporter, torch_runtime_status
from transformer_training_parity_fixture import validate_training_parity_fixture


TORCH_TRAINING_PARITY_CANDIDATE_KIND = (
    "transformer_torch_training_parity_candidate"
)
TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION = 1
TORCH_TRAINING_IMPLEMENTATION_STATUS = "training_not_implemented"


def build_torch_training_parity_candidate(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter = import_module,
    requested_device: str = "cpu",
    requested_dtype: str = "float32",
) -> dict[str, Any]:
    """Build a PyTorch training candidate artifact without claiming parity."""

    validate_training_parity_fixture(fixture)
    runtime = torch_runtime_status(
        importer=importer,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
    )
    outputs = _candidate_outputs(fixture=fixture, runtime=runtime)
    return {
        "schema_version": TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION,
        "kind": TORCH_TRAINING_PARITY_CANDIDATE_KIND,
        "fixture_id": fixture["fixture_id"],
        "implementation_status": outputs["implementation_status"],
        "backend": transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            seed=fixture["reference_backend"]["seed"],
            tokenizer_type=fixture["tokenizer"]["tokenizer_type"],
            corpus_hash=fixture["reference_backend"]["corpus_hash"],
            tokenizer_manifest_hash=fixture["tokenizer"].get(
                "tokenizer_manifest_hash"
            ),
            device=runtime["device"],
            dtype=runtime["dtype"],
            parity_status=outputs["parity_status"],
        ),
        "runtime": runtime,
        "model_config": dict(fixture["model_config"]),
        "tokenizer": dict(fixture["tokenizer"]),
        "optimizer_config": dict(fixture["optimizer_config"]),
        "parameter_manifest": dict(fixture["parameter_manifest"]),
        "training_case": outputs["training_case"],
    }


def _candidate_outputs(
    *,
    fixture: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if not runtime["available"]:
        return {
            "implementation_status": "runtime_unavailable",
            "parity_status": "failed",
            "training_case": _case_stub(
                fixture["training_case"],
                status="blocked",
                reason="pytorch runtime is unavailable",
            ),
        }
    if not runtime["dtype_available"]:
        return {
            "implementation_status": "dtype_unavailable",
            "parity_status": "pending",
            "training_case": _case_stub(
                fixture["training_case"],
                status="pending",
                reason="requested pytorch dtype is unavailable",
            ),
        }
    return {
        "implementation_status": TORCH_TRAINING_IMPLEMENTATION_STATUS,
        "parity_status": "pending",
        "training_case": _case_stub(
            fixture["training_case"],
            status="pending",
            reason="pytorch training parity is not implemented yet",
        ),
    }


def _case_stub(
    case: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "status": status,
        "reason": reason,
        "context": list(case["context"]),
        "target": case["target"],
        "learning_rate": case["learning_rate"],
        "steps": case["steps"],
    }
