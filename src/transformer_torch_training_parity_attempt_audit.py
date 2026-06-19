"""Compact audit results for written PyTorch training parity attempts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from transformer_torch_training_parity_attempt_audit_validation import (
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
    validate_torch_training_parity_attempt_audit,
)
from transformer_torch_training_parity_attempt_reader import (
    TORCH_TRAINING_PARITY_ATTEMPT_FILES,
    load_torch_training_parity_attempt_artifact_set,
)


def build_torch_training_parity_attempt_audit(output_dir: Path) -> dict[str, Any]:
    """Return a JSON-safe verification result for a written attempt directory."""

    try:
        artifacts = load_torch_training_parity_attempt_artifact_set(output_dir)
    except (OSError, ValueError) as exc:
        return _validated(_invalid_audit(output_dir, exc))
    return _validated(_valid_audit(output_dir, artifacts["attempt"]))


def _base_audit(output_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
        "output_dir": str(output_dir),
        "artifact_files": dict(TORCH_TRAINING_PARITY_ATTEMPT_FILES),
    }


def _valid_audit(output_dir: Path, attempt: dict[str, Any]) -> dict[str, Any]:
    next_requirements = attempt["next_requirements"]
    promotion_gate = attempt["training_backend_promotion_gate"]
    runtime = attempt["runtime"]
    return {
        **_base_audit(output_dir),
        "status": "artifact_set_valid",
        "passed": True,
        "fixture_id": attempt["fixture_id"],
        "attempt_status": attempt["status"],
        "attempt_passed": attempt["passed"],
        "runtime_status": runtime["status"],
        "parity_attempt_allowed": runtime["parity_attempt_allowed"],
        "next_requirements_stage": next_requirements["stage"],
        "next_requirements_status": next_requirements["status"],
        "next_actions": list(next_requirements["next_actions"]),
        "training_backend_promotion_status": promotion_gate["status"],
        "promoted_training_backend": attempt["promoted_training_backend"],
        "artifact_hash_algorithm": attempt["artifact_hash_algorithm"],
        "artifact_hashes": dict(attempt["artifact_hashes"]),
        "evidence_hashes": _evidence_hashes(attempt),
    }


def _invalid_audit(output_dir: Path, exc: Exception) -> dict[str, Any]:
    return {
        **_base_audit(output_dir),
        "status": "artifact_set_invalid",
        "passed": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _evidence_hashes(attempt: dict[str, Any]) -> dict[str, str]:
    return {
        "runtime_report": attempt["runtime"]["runtime_report_sha256"],
        "candidate": attempt["candidate"]["candidate_sha256"],
        "training_replay_parity_gate": attempt["training_replay_parity_gate"][
            "training_replay_parity_gate_sha256"
        ],
        "training_parity_report": attempt["training_parity_report"][
            "training_parity_report_sha256"
        ],
    }


def _validated(audit: dict[str, Any]) -> dict[str, Any]:
    validate_torch_training_parity_attempt_audit(audit)
    return audit
