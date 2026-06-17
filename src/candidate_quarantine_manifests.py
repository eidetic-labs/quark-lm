"""Candidate quarantine manifest assembly and persistence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from candidate_quarantine_records import (
    ALLOWED_TRANSITIONS,
    CANDIDATE_STATES,
    SCHEMA_VERSION,
    TRAINING_ELIGIBLE_STATES,
    validate_candidate_record,
)


def _candidate_counts(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_state = Counter(candidate["state"] for candidate in candidates)
    by_type = Counter(candidate["candidate_type"] for candidate in candidates)
    eligible = sum(
        count
        for state, count in by_state.items()
        if state in TRAINING_ELIGIBLE_STATES
    )
    return {
        "total": len(candidates),
        "by_state": dict(sorted(by_state.items())),
        "by_type": dict(sorted(by_type.items())),
        "training_eligible": eligible,
        "not_training_eligible": len(candidates) - eligible,
    }


def build_candidate_quarantine_manifest(
    component: str,
    run_id: str,
    candidates: list[dict[str, Any]] | None = None,
    candidate_sources: list[str] | None = None,
) -> dict[str, Any]:
    records = list(candidates or [])
    for record in records:
        validate_candidate_record(record)
    counts = _candidate_counts(records)
    if counts["total"] == 0:
        status = "empty_no_candidates"
    elif counts["not_training_eligible"]:
        status = "contains_quarantined_candidates"
    else:
        status = "all_candidates_admitted"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_quarantine_manifest",
        "component": component,
        "run_id": run_id,
        "status": status,
        "allowed_states": list(CANDIDATE_STATES),
        "allowed_transitions": {
            state: list(next_states)
            for state, next_states in sorted(ALLOWED_TRANSITIONS.items())
        },
        "candidate_sources": list(candidate_sources or []),
        "candidate_counts": counts,
        "training_policy": {
            "candidate_records_are_training_data": False,
            "training_eligible_states": list(TRAINING_ELIGIBLE_STATES),
            "rule": "Candidate records are not training data until admitted into the ledgered corpus and converted into curriculum lessons.",
        },
        "candidates": records,
    }


def validate_candidate_quarantine_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate quarantine schema_version")
    if manifest.get("kind") != "candidate_quarantine_manifest":
        raise ValueError("kind must be candidate_quarantine_manifest")
    for field_name in ("component", "run_id", "status"):
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(manifest.get("candidates"), list):
        raise ValueError("candidates must be a list")
    for record in manifest["candidates"]:
        validate_candidate_record(record)


def candidate_quarantine_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_candidate_quarantine_manifest(manifest)
    counts = manifest["candidate_counts"]
    return {
        "status": manifest["status"],
        "candidate_count": counts["total"],
        "by_state": dict(counts["by_state"]),
        "by_type": dict(counts["by_type"]),
        "training_eligible_count": counts["training_eligible"],
        "not_training_eligible_count": counts["not_training_eligible"],
        "candidate_records_are_training_data": False,
        "training_policy": manifest["training_policy"]["rule"],
    }


def write_candidate_quarantine(path: Path, manifest: dict[str, Any]) -> None:
    validate_candidate_quarantine_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
