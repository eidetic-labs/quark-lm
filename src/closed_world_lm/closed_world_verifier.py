"""Deterministic verifier checks for closed-world training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .candidate_quarantine import (
    TRAINING_ELIGIBLE_STATES,
    validate_candidate_quarantine_manifest,
    validate_candidate_record,
)
from .respond import CorpusResponder


SCHEMA_VERSION = 1
REPORT_KIND = "closed_world_verifier_report"
PASS = "passed"
FAIL = "failed"


def verifier_check(
    name: str,
    passed: bool,
    rule: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": PASS if passed else FAIL,
        "passed": bool(passed),
        "rule": rule,
        "details": dict(details or {}),
    }


def verifier_report(
    component: str,
    run_id: str,
    subject_kind: str,
    checks: list[dict[str, Any]],
    subject_path: Path | str | None = None,
    related_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    failed = [check["name"] for check in checks if not check.get("passed")]
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "component": component,
        "run_id": run_id,
        "subject_kind": subject_kind,
        "subject_path": str(subject_path) if subject_path is not None else None,
        "passed": not failed,
        "failed_checks": failed,
        "summary": {
            "check_count": len(checks),
            "passed_count": len(checks) - len(failed),
            "failed_count": len(failed),
        },
        "uses_external_model": False,
        "verifier_type": "deterministic_closed_world",
        "checks": checks,
        "related_reports": list(related_reports or []),
    }
    validate_verifier_report(report)
    return report


def verifier_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    validate_verifier_report(report)
    return {
        "status": PASS if report["passed"] else FAIL,
        "passed": report["passed"],
        "failed_checks": list(report["failed_checks"]),
        "check_count": report["summary"]["check_count"],
        "passed_count": report["summary"]["passed_count"],
        "failed_count": report["summary"]["failed_count"],
        "uses_external_model": False,
        "verifier_type": report["verifier_type"],
    }


def validate_verifier_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported verifier schema_version")
    if report.get("kind") != REPORT_KIND:
        raise ValueError(f"kind must be {REPORT_KIND}")
    for field_name in ("component", "run_id", "subject_kind", "verifier_type"):
        value = report.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(report.get("uses_external_model"), bool):
        raise ValueError("uses_external_model must be a bool")
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise ValueError("checks must be a list")
    for check in checks:
        _validate_check(check)
    failed = [check["name"] for check in checks if not check["passed"]]
    if report.get("failed_checks") != failed:
        raise ValueError("failed_checks must match failed checks")
    if report.get("passed") != (not failed):
        raise ValueError("passed must match checks")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("summary must be a dict")
    if summary.get("check_count") != len(checks):
        raise ValueError("summary check_count must match checks")
    if summary.get("failed_count") != len(failed):
        raise ValueError("summary failed_count must match failed checks")
    if summary.get("passed_count") != len(checks) - len(failed):
        raise ValueError("summary passed_count must match passed checks")
    related = report.get("related_reports")
    if not isinstance(related, list):
        raise ValueError("related_reports must be a list")


def _validate_check(check: dict[str, Any]) -> None:
    for field_name in ("name", "status", "rule"):
        value = check.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"check {field_name} must be a non-empty string")
    if not isinstance(check.get("passed"), bool):
        raise ValueError("check passed must be a bool")
    expected_status = PASS if check["passed"] else FAIL
    if check["status"] != expected_status:
        raise ValueError("check status must match passed")
    if not isinstance(check.get("details"), dict):
        raise ValueError("check details must be a dict")


def write_verifier_report(path: Path, report: dict[str, Any]) -> None:
    validate_verifier_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def attach_verifier_summary(
    training_plan: dict[str, Any],
    report: dict[str, Any],
    verifier_path: Path,
) -> dict[str, Any]:
    updated = dict(training_plan)
    updated["closed_world_verifier"] = {
        "status": "written",
        "path": str(verifier_path),
        "summary": verifier_report_summary(report),
    }
    return updated


def verify_candidate_record(
    record: dict[str, Any],
    responder: CorpusResponder | None = None,
) -> dict[str, Any]:
    validation_error: str | None = None
    try:
        validate_candidate_record(record)
    except ValueError as exc:
        validation_error = str(exc)

    candidate_type = record.get("candidate_type")
    prompt = record.get("prompt", "")
    target = record.get("target", "")
    proposal = record.get("proposal", "")
    state = record.get("state", "")
    admission_id = record.get("admission_id")
    checks = [
        verifier_check(
            "candidate_record_valid",
            validation_error is None,
            "Candidate records must satisfy the quarantine schema.",
            {"error": validation_error},
        ),
        verifier_check(
            "source_label_present",
            isinstance(record.get("source"), str) and bool(record.get("source", "").strip()),
            "Every candidate must retain a non-empty origin label.",
            {"source": record.get("source")},
        ),
    ]

    payload_present = _candidate_payload_present(candidate_type, prompt, target, proposal)
    checks.append(
        verifier_check(
            "candidate_payload_present",
            payload_present,
            "A candidate must carry the prompt, target, or proposal fields required by its type.",
            {
                "candidate_type": candidate_type,
                "has_prompt": bool(prompt),
                "has_target": bool(target),
                "has_proposal": bool(proposal),
            },
        )
    )

    if prompt and target:
        if responder is None:
            checks.append(
                verifier_check(
                    "exact_answer_consistency",
                    True,
                    "Prompt-target candidates can be checked against a closed-world responder before admission.",
                    {"reason": "responder_missing"},
                )
            )
        else:
            answer = responder.answer_prompt(prompt)
            checks.append(
                verifier_check(
                    "exact_answer_consistency",
                    answer == target,
                    "Candidate targets must match the deterministic responder trained from admitted text.",
                    {"answer": answer, "target": target},
                )
            )

    eligible = state in TRAINING_ELIGIBLE_STATES
    checks.append(
        verifier_check(
            "training_eligible_state_has_admission",
            not eligible or (isinstance(admission_id, str) and bool(admission_id.strip())),
            "Training-eligible candidate states must link to a ledger admission id.",
            {"state": state, "admission_id": admission_id},
        )
    )

    return verifier_report(
        component="candidate_quarantine",
        run_id=str(record.get("candidate_id", "candidate")),
        subject_kind="candidate_record",
        checks=checks,
    )


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

    candidates = manifest.get("candidates") if isinstance(manifest.get("candidates"), list) else []
    counts = manifest.get("candidate_counts") if isinstance(manifest.get("candidate_counts"), dict) else {}
    policy = manifest.get("training_policy") if isinstance(manifest.get("training_policy"), dict) else {}
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
            {"candidate_records_are_training_data": policy.get("candidate_records_are_training_data")},
        ),
        verifier_check(
            "candidate_counts_consistent",
            counts.get("total") == len(candidates)
            and counts.get("training_eligible", 0) + counts.get("not_training_eligible", 0)
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
        related_reports=[verifier_report_summary(report) for report in candidate_reports],
    )


def verify_training_plan(
    training_plan: dict[str, Any],
    corpus_hygiene: dict[str, Any] | None = None,
    candidate_quarantine: dict[str, Any] | None = None,
    subject_path: Path | str | None = None,
    verifier_path: Path | str | None = None,
) -> dict[str, Any]:
    boundary = (
        training_plan.get("data_boundary")
        if isinstance(training_plan.get("data_boundary"), dict)
        else {}
    )
    candidate_policy = (
        training_plan.get("candidate_policy")
        if isinstance(training_plan.get("candidate_policy"), dict)
        else {}
    )
    candidate_quarantine_info = (
        candidate_policy.get("candidate_quarantine")
        if isinstance(candidate_policy.get("candidate_quarantine"), dict)
        else {}
    )
    planned_artifacts = training_plan.get("planned_artifacts")
    if not isinstance(planned_artifacts, list):
        planned_artifacts = []
    quarantine_report = (
        verify_candidate_quarantine_manifest(candidate_quarantine)
        if candidate_quarantine is not None
        else None
    )
    train_eval_overlap = (
        corpus_hygiene.get("train_eval_overlap")
        if isinstance(corpus_hygiene, dict)
        and isinstance(corpus_hygiene.get("train_eval_overlap"), dict)
        else None
    )
    checks = [
        verifier_check(
            "training_plan_valid",
            training_plan.get("schema_version") == 1
            and training_plan.get("kind") == "training_plan"
            and isinstance(training_plan.get("component"), str)
            and isinstance(training_plan.get("run_id"), str),
            "Training plans must be structured artifacts with component and run identity.",
            {
                "schema_version": training_plan.get("schema_version"),
                "kind": training_plan.get("kind"),
            },
        ),
        verifier_check(
            "closed_world_data_boundary",
            all(
                boundary.get(flag) is False
                for flag in (
                    "pretrained_weights",
                    "pretrained_tokenizer",
                    "external_embeddings",
                    "unledgered_training_text",
                )
            ),
            "Training plans must reject pretrained weights, pretrained tokenizers, external embeddings, and unledgered training text.",
            dict(boundary),
        ),
        verifier_check(
            "candidate_records_excluded",
            candidate_policy.get("candidate_records_are_training_data") is False,
            "Candidate records must not be treated as training data.",
            {
                "candidate_records_are_training_data": candidate_policy.get(
                    "candidate_records_are_training_data"
                )
            },
        ),
        verifier_check(
            "no_candidate_examples_in_training",
            candidate_policy.get("candidate_examples", 0) == 0
            and candidate_policy.get("status") != "training_examples_contain_candidates",
            "Training examples must not include candidate-sourced examples.",
            {
                "candidate_examples": candidate_policy.get("candidate_examples"),
                "status": candidate_policy.get("status"),
            },
        ),
        verifier_check(
            "candidate_quarantine_declared",
            isinstance(candidate_quarantine_info.get("path"), str)
            and bool(candidate_quarantine_info.get("path"))
            and isinstance(candidate_quarantine_info.get("summary"), dict),
            "Training plans must name the candidate quarantine artifact and summary.",
            {
                "path": candidate_quarantine_info.get("path"),
                "has_summary": isinstance(candidate_quarantine_info.get("summary"), dict),
            },
        ),
        verifier_check(
            "candidate_quarantine_passes",
            quarantine_report is not None and quarantine_report["passed"],
            "The candidate quarantine manifest must pass deterministic verifier checks.",
            (
                verifier_report_summary(quarantine_report)
                if quarantine_report is not None
                else {"status": "missing"}
            ),
        ),
        verifier_check(
            "protected_train_eval_overlap_passes",
            train_eval_overlap is not None and train_eval_overlap.get("passed") is True,
            "Protected eval prompts must not overlap training examples or training text.",
            (
                {
                    "protected_prompt_overlap_count": train_eval_overlap.get(
                        "protected_prompt_overlap_count"
                    ),
                    "protected_train_text_prompt_overlap_count": train_eval_overlap.get(
                        "protected_train_text_prompt_overlap_count"
                    ),
                }
                if train_eval_overlap is not None
                else {"status": "missing"}
            ),
        ),
        verifier_check(
            "verifier_artifact_planned",
            verifier_path is None or str(verifier_path) in planned_artifacts,
            "Runs that write verifier evidence must declare it before training.",
            {"verifier_path": str(verifier_path) if verifier_path is not None else None},
        ),
    ]
    return verifier_report(
        component=str(training_plan.get("component", "training_plan")),
        run_id=str(training_plan.get("run_id", "training_plan")),
        subject_kind="training_plan",
        subject_path=subject_path,
        checks=checks,
        related_reports=(
            [verifier_report_summary(quarantine_report)]
            if quarantine_report is not None
            else []
        ),
    )


def _candidate_payload_present(
    candidate_type: Any,
    prompt: Any,
    target: Any,
    proposal: Any,
) -> bool:
    has_prompt_target = isinstance(prompt, str) and bool(prompt.strip()) and isinstance(
        target, str
    ) and bool(target.strip())
    has_proposal = isinstance(proposal, str) and bool(proposal.strip())
    if candidate_type in ("lesson", "probe"):
        return has_prompt_target
    if candidate_type in ("repair_proposal", "diagnosis", "memory"):
        return has_proposal or has_prompt_target
    return False
