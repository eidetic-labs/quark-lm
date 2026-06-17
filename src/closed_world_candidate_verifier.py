"""Compatibility exports for candidate quarantine verifier checks."""

from __future__ import annotations

from closed_world_candidate_manifest_verifier import (
    verify_candidate_quarantine_manifest,
)
from closed_world_candidate_record_verifier import verify_candidate_record

__all__ = [
    "verify_candidate_quarantine_manifest",
    "verify_candidate_record",
]
