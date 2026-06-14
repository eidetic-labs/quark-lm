"""Candidate quarantine artifacts for closed-world self-improvement."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CANDIDATE_TYPES = ("lesson", "probe", "repair_proposal", "memory", "diagnosis")
CANDIDATE_STATES = (
    "proposed",
    "quarantined",
    "needs_human_review",
    "verified",
    "rejected",
    "admitted",
    "trained",
    "promoted",
)
TRAINING_ELIGIBLE_STATES = ("admitted", "trained", "promoted")
ALLOWED_TRANSITIONS = {
    "proposed": ("quarantined", "needs_human_review", "rejected"),
    "quarantined": ("verified", "needs_human_review", "rejected"),
    "needs_human_review": ("quarantined", "verified", "rejected"),
    "verified": ("admitted", "needs_human_review", "rejected"),
    "admitted": ("trained",),
    "trained": ("promoted",),
    "promoted": (),
    "rejected": (),
}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def candidate_id_from_parts(
    candidate_type: str,
    source: str,
    prompt: str = "",
    target: str = "",
    proposal: str = "",
) -> str:
    payload = {
        "candidate_type": candidate_type,
        "prompt": prompt,
        "proposal": proposal,
        "source": source,
        "target": target,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{candidate_type}-{digest}"


def candidate_record(
    candidate_type: str,
    source: str,
    prompt: str = "",
    target: str = "",
    proposal: str = "",
    state: str = "quarantined",
    evidence: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
    admission_id: str | None = None,
    created_at: str | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "candidate_record",
        "candidate_id": candidate_id
        or candidate_id_from_parts(candidate_type, source, prompt, target, proposal),
        "candidate_type": candidate_type,
        "state": state,
        "source": source,
        "prompt": prompt,
        "target": target,
        "proposal": proposal,
        "evidence": list(evidence or []),
        "notes": list(notes or []),
        "admission_id": admission_id,
        "created_at": created_at or utc_now_iso(),
        "transitions": [],
    }
    validate_candidate_record(record)
    return record


def validate_candidate_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate record schema_version")
    if record.get("kind") != "candidate_record":
        raise ValueError("kind must be candidate_record")
    for field_name in ("candidate_id", "source", "created_at"):
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if record.get("candidate_type") not in CANDIDATE_TYPES:
        raise ValueError(f"candidate_type must be one of {list(CANDIDATE_TYPES)}")
    if record.get("state") not in CANDIDATE_STATES:
        raise ValueError(f"state must be one of {list(CANDIDATE_STATES)}")
    for field_name in ("prompt", "target", "proposal"):
        if not isinstance(record.get(field_name, ""), str):
            raise ValueError(f"{field_name} must be a string")
    if not isinstance(record.get("evidence"), list):
        raise ValueError("evidence must be a list")
    if not isinstance(record.get("notes"), list):
        raise ValueError("notes must be a list")
    if not isinstance(record.get("transitions"), list):
        raise ValueError("transitions must be a list")
    admission_id = record.get("admission_id")
    if admission_id is not None and not isinstance(admission_id, str):
        raise ValueError("admission_id must be a string or null")


def transition_candidate(
    record: dict[str, Any],
    state: str,
    evidence: list[dict[str, Any]] | None = None,
    note: str | None = None,
    admission_id: str | None = None,
    transitioned_at: str | None = None,
) -> dict[str, Any]:
    validate_candidate_record(record)
    current = record["state"]
    if state not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"cannot transition candidate from {current!r} to {state!r}")
    updated = {
        **record,
        "state": state,
        "evidence": list(record.get("evidence", [])),
        "notes": list(record.get("notes", [])),
        "transitions": list(record.get("transitions", [])),
    }
    transition = {
        "from_state": current,
        "to_state": state,
        "transitioned_at": transitioned_at or utc_now_iso(),
        "evidence": list(evidence or []),
        "note": note,
    }
    updated["transitions"].append(transition)
    updated["evidence"].extend(evidence or [])
    if note:
        updated["notes"].append(note)
    if admission_id is not None:
        updated["admission_id"] = admission_id
    validate_candidate_record(updated)
    return updated


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
