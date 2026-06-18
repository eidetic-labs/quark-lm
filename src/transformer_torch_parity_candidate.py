"""PyTorch parity-candidate artifacts for transformer backend experiments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from transformer_backend_parity_validation import validate_backend_parity_fixture
from transformer_backend_policy import PYTORCH_BACKEND, transformer_backend_metadata
from transformer_torch_runtime import TorchImporter, torch_runtime_status


TORCH_PARITY_CANDIDATE_KIND = "transformer_torch_backend_parity_candidate"
TORCH_PARITY_CANDIDATE_SCHEMA_VERSION = 1
TORCH_PARITY_IMPLEMENTATION_STATUS = "skeleton"


def build_torch_backend_parity_candidate(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter = import_module,
    requested_device: str = "cpu",
    requested_dtype: str = "float32",
) -> dict[str, Any]:
    """Build a PyTorch candidate artifact without requiring PyTorch to be installed."""

    validate_backend_parity_fixture(fixture)
    runtime = torch_runtime_status(
        importer=importer,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
    )
    return {
        "schema_version": TORCH_PARITY_CANDIDATE_SCHEMA_VERSION,
        "kind": TORCH_PARITY_CANDIDATE_KIND,
        "fixture_id": fixture["fixture_id"],
        "implementation_status": TORCH_PARITY_IMPLEMENTATION_STATUS,
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
            parity_status="pending" if runtime["available"] else "failed",
        ),
        "runtime": runtime,
        "model_config": dict(fixture["model_config"]),
        "tokenizer": dict(fixture["tokenizer"]),
        "forward_cases": [
            _case_stub(case, runtime)
            for case in fixture["forward_cases"]
        ],
        "generation_cases": [
            _case_stub(case, runtime)
            for case in fixture.get("generation_cases", [])
        ],
    }


def _case_stub(case: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    reason = (
        "pytorch runtime is unavailable"
        if not runtime["available"]
        else "pytorch parity math is not implemented yet"
    )
    return {
        "case_id": case["case_id"],
        "status": "blocked" if not runtime["available"] else "pending",
        "reason": reason,
    }
