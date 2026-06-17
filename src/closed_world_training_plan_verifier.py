"""Closed-world verifier checks for training plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from closed_world_candidate_verifier import verify_candidate_quarantine_manifest
from closed_world_verifier_reports import (
    verifier_check,
    verifier_report,
    verifier_report_summary,
)


def verify_training_plan(
    training_plan: dict[str, Any],
    corpus_hygiene: dict[str, Any] | None = None,
    candidate_quarantine: dict[str, Any] | None = None,
    subject_path: Path | str | None = None,
    verifier_path: Path | str | None = None,
) -> dict[str, Any]:
    boundary = _dict_field(training_plan, "data_boundary")
    candidate_policy = _dict_field(training_plan, "candidate_policy")
    candidate_quarantine_info = _dict_field(
        candidate_policy,
        "candidate_quarantine",
    )
    planned_artifacts = training_plan.get("planned_artifacts")
    if not isinstance(planned_artifacts, list):
        planned_artifacts = []
    quarantine_report = (
        verify_candidate_quarantine_manifest(candidate_quarantine)
        if candidate_quarantine is not None
        else None
    )
    train_eval_overlap = _train_eval_overlap(corpus_hygiene)
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
            and candidate_policy.get("status")
            != "training_examples_contain_candidates",
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
                "has_summary": isinstance(
                    candidate_quarantine_info.get("summary"),
                    dict,
                ),
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
            train_eval_overlap is not None
            and train_eval_overlap.get("passed") is True,
            "Protected eval prompts must not overlap training examples or training text.",
            _overlap_details(train_eval_overlap),
        ),
        verifier_check(
            "verifier_artifact_planned",
            verifier_path is None or str(verifier_path) in planned_artifacts,
            "Runs that write verifier evidence must declare it before training.",
            {
                "verifier_path": (
                    str(verifier_path) if verifier_path is not None else None
                )
            },
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


def _dict_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _train_eval_overlap(
    corpus_hygiene: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(corpus_hygiene, dict):
        return None
    overlap = corpus_hygiene.get("train_eval_overlap")
    return overlap if isinstance(overlap, dict) else None


def _overlap_details(train_eval_overlap: dict[str, Any] | None) -> dict[str, Any]:
    if train_eval_overlap is None:
        return {"status": "missing"}
    return {
        "protected_prompt_overlap_count": train_eval_overlap.get(
            "protected_prompt_overlap_count"
        ),
        "protected_train_text_prompt_overlap_count": train_eval_overlap.get(
            "protected_train_text_prompt_overlap_count"
        ),
    }
