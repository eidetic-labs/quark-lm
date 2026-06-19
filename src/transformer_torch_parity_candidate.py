"""PyTorch parity-candidate artifacts for transformer backend experiments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from transformer_backend_parity_validation import validate_backend_parity_fixture
from transformer_backend_policy import PYTORCH_BACKEND, transformer_backend_metadata
from transformer_torch_minimal_forward import torch_minimal_parity_outputs
from transformer_torch_runtime import TorchImporter
from transformer_torch_runtime_report import build_torch_runtime_report


TORCH_PARITY_CANDIDATE_KIND = "transformer_torch_backend_parity_candidate"
TORCH_PARITY_CANDIDATE_SCHEMA_VERSION = 1
TORCH_PARITY_IMPLEMENTATION_STATUS = "minimal_forward"


def build_torch_backend_parity_candidate(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter = import_module,
    requested_device: str = "cpu",
    requested_dtype: str = "float32",
) -> dict[str, Any]:
    """Build a PyTorch candidate artifact without requiring PyTorch to be installed."""

    validate_backend_parity_fixture(fixture)
    runtime_report = build_torch_runtime_report(
        importer=importer,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
    )
    runtime = runtime_report["runtime"]
    outputs = _candidate_outputs(
        fixture=fixture,
        importer=importer,
        runtime=runtime,
    )
    return {
        "schema_version": TORCH_PARITY_CANDIDATE_SCHEMA_VERSION,
        "kind": TORCH_PARITY_CANDIDATE_KIND,
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
        "runtime_report": runtime_report,
        "model_config": dict(fixture["model_config"]),
        "tokenizer": dict(fixture["tokenizer"]),
        "forward_cases": outputs["forward_cases"],
        "generation_cases": outputs["generation_cases"],
    }


def _candidate_outputs(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if not runtime["available"]:
        return {
            "implementation_status": "runtime_unavailable",
            "parity_status": "failed",
            "forward_cases": [
                _blocked_case(case["case_id"])
                for case in fixture["forward_cases"]
            ],
            "generation_cases": [
                _blocked_case(case["case_id"])
                for case in fixture.get("generation_cases", [])
            ],
        }
    if not runtime["dtype_available"]:
        return {
            "implementation_status": "dtype_unavailable",
            "parity_status": "pending",
            "forward_cases": [
                _pending_case(case["case_id"], "requested pytorch dtype is unavailable")
                for case in fixture["forward_cases"]
            ],
            "generation_cases": [
                _pending_case(case["case_id"], "requested pytorch dtype is unavailable")
                for case in fixture.get("generation_cases", [])
            ],
        }
    torch = importer("torch")
    return torch_minimal_parity_outputs(
        fixture=fixture,
        torch=torch,
        runtime=runtime,
    )


def _blocked_case(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "blocked",
        "reason": "pytorch runtime is unavailable",
    }


def _pending_case(case_id: str, reason: str) -> dict[str, Any]:
    return {"case_id": case_id, "status": "pending", "reason": reason}
