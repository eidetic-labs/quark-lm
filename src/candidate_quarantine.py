"""Compatibility exports for candidate quarantine artifacts."""

from __future__ import annotations

from candidate_quarantine_manifests import (
    build_candidate_quarantine_manifest,
    candidate_quarantine_summary,
    validate_candidate_quarantine_manifest,
    write_candidate_quarantine,
)
from candidate_quarantine_records import (
    ALLOWED_TRANSITIONS,
    CANDIDATE_STATES,
    CANDIDATE_TYPES,
    SCHEMA_VERSION,
    TRAINING_ELIGIBLE_STATES,
    candidate_id_from_parts,
    candidate_record,
    transition_candidate,
    utc_now_iso,
    validate_candidate_record,
)


__all__ = [
    "ALLOWED_TRANSITIONS",
    "CANDIDATE_STATES",
    "CANDIDATE_TYPES",
    "SCHEMA_VERSION",
    "TRAINING_ELIGIBLE_STATES",
    "build_candidate_quarantine_manifest",
    "candidate_id_from_parts",
    "candidate_quarantine_summary",
    "candidate_record",
    "transition_candidate",
    "utc_now_iso",
    "validate_candidate_quarantine_manifest",
    "validate_candidate_record",
    "write_candidate_quarantine",
]
