"""Candidate quarantine record lifecycle helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
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
