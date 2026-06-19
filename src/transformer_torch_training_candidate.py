"""PyTorch training parity candidate artifacts for transformer experiments."""

from __future__ import annotations

import copy
from importlib import import_module
from typing import Any

from transformer_backend_policy import PYTORCH_BACKEND, transformer_backend_metadata
from transformer_torch_runtime import TorchImporter, torch_runtime_status
from transformer_torch_training_candidate_probes import (
    build_torch_training_probe_artifacts,
)
from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
    build_torch_training_replay_parity_gate,
)
from transformer_torch_training_readiness import (
    TORCH_TRAINING_READY_STATUS,
    build_torch_training_readiness,
)
from transformer_training_parity_fixture import validate_training_parity_fixture


TORCH_TRAINING_PARITY_CANDIDATE_KIND = (
    "transformer_torch_training_parity_candidate"
)
TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION = 1
TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS = "training_runtime_incomplete"
TORCH_TRAINING_REPLAY_PARITY_STATUS = TORCH_TRAINING_REPLAY_PENDING_STATUS


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
    readiness = build_torch_training_readiness(
        fixture=fixture,
        runtime=runtime,
        importer=importer,
    )
    probes = build_torch_training_probe_artifacts(
        fixture=fixture,
        importer=importer,
        readiness=readiness,
        runtime=runtime,
    )
    gate = build_torch_training_replay_parity_gate(
        runtime=runtime,
        readiness=readiness,
        probes=probes,
    )
    outputs = _candidate_outputs(
        fixture=fixture,
        runtime=runtime,
        readiness=readiness,
        gate=gate,
    )
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
        "optimizer_step_contract": copy.deepcopy(
            fixture["optimizer_step_contract"]
        ),
        "training_readiness": readiness,
        **probes,
        "training_replay_parity_gate": gate,
        "training_case": outputs["training_case"],
    }


def _candidate_outputs(
    *,
    fixture: dict[str, Any],
    runtime: dict[str, Any],
    readiness: dict[str, Any],
    gate: dict[str, Any],
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
    if readiness["status"] != TORCH_TRAINING_READY_STATUS:
        return {
            "implementation_status": TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
            "parity_status": "pending",
            "training_case": _case_stub(
                fixture["training_case"],
                status="pending",
                reason="pytorch training runtime is missing required capabilities",
            ),
        }
    if gate["status"] == TORCH_TRAINING_REPLAY_MATCHED_STATUS:
        return {
            "implementation_status": TORCH_TRAINING_REPLAY_MATCHED_STATUS,
            "parity_status": "matched",
            "training_case": _matched_case(fixture["training_case"]),
        }
    return {
        "implementation_status": TORCH_TRAINING_REPLAY_PENDING_STATUS,
        "parity_status": "pending",
        "training_case": _case_stub(
            fixture["training_case"],
            status="pending",
            reason=gate["reason"],
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


def _matched_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **copy.deepcopy(case),
        "status": "matched",
        "reason": "pytorch replay parity gates matched scalar training evidence",
        "evidence_source": "training_replay_parity_gate",
        "promoted_training_backend": False,
    }
