"""Closed-world verifier checks for candidate quarantine manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from candidate_quarantine import (
    TRAINING_ELIGIBLE_STATES,
    validate_candidate_quarantine_manifest,
)
from closed_world_candidate_record_verifier import verify_candidate_record
from closed_world_verifier_reports import (
    verifier_check,
    verifier_report,
    verifier_report_summary,
)
from corpus_responder import CorpusResponder


def verify_candidate_quarantine_manifest(
    manifest: dict[str, Any],
    responder: CorpusResponder | None = None,
    subject_path: Path | str | None = None,
) -> dict[str, Any]:
    validation_error: str | None = None
    try:
        validate_candidate_quarantine_manifest(manifest)
    except ValueError as exc:
        validation_error = str(exc)

    candidates = (
        manifest.get("candidates")
        if isinstance(manifest.get("candidates"), list)
        else []
    )
    counts = (
        manifest.get("candidate_counts")
        if isinstance(manifest.get("candidate_counts"), dict)
        else {}
    )
    policy = (
        manifest.get("training_policy")
        if isinstance(manifest.get("training_policy"), dict)
        else {}
    )
    candidate_reports = [
        verify_candidate_record(candidate, responder=responder)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    eligible_without_admission = [
        candidate.get("candidate_id")
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("state") in TRAINING_ELIGIBLE_STATES
        and not candidate.get("admission_id")
    ]
    checks = [
        verifier_check(
            "candidate_quarantine_manifest_valid",
            validation_error is None,
            "Candidate quarantine manifests must satisfy the quarantine schema.",
            {"error": validation_error},
        ),
        verifier_check(
            "candidate_records_excluded_from_training",
            policy.get("candidate_records_are_training_data") is False,
            "Candidate records are not training data until converted into admitted curriculum lessons.",
            {
                "candidate_records_are_training_data": policy.get(
                    "candidate_records_are_training_data"
                )
            },
        ),
        verifier_check(
            "candidate_counts_consistent",
            counts.get("total") == len(candidates)
            and counts.get("training_eligible", 0)
            + counts.get("not_training_eligible", 0)
            == len(candidates),
            "Candidate manifest counts must match the candidate list.",
            {"record_count": len(candidates), "counts": counts},
        ),
        verifier_check(
            "training_eligible_candidates_have_admissions",
            not eligible_without_admission,
            "Training-eligible candidates must reference a ledger admission id.",
            {"candidate_ids": eligible_without_admission},
        ),
        verifier_check(
            "candidate_records_pass_verifier",
            all(report["passed"] for report in candidate_reports),
            "Every candidate record must pass deterministic closed-world checks.",
            {
                "candidate_count": len(candidate_reports),
                "failed_candidate_ids": [
                    report["run_id"]
                    for report in candidate_reports
                    if not report["passed"]
                ],
            },
        ),
    ]
    return verifier_report(
        component=str(manifest.get("component", "candidate_quarantine")),
        run_id=str(manifest.get("run_id", "candidate_quarantine")),
        subject_kind="candidate_quarantine_manifest",
        subject_path=subject_path,
        checks=checks,
        related_reports=[
            verifier_report_summary(report) for report in candidate_reports
        ],
    )
