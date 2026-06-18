"""Training-plan artifact assembly for closed-world runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from corpus_eval_summary import eval_set_counts
from corpus_example_summary import rare_profile_coverage, source_mixture


def build_training_plan(
    component: str,
    run_id: str,
    train_text_path: Path,
    corpus_dir: Path,
    eval_paths: list[Path],
    training_examples: list[Any],
    training_pool: list[Any],
    hygiene_path: Path,
    planned_artifacts: list[Path] | None = None,
    replay_plan_path: Path | None = None,
    candidate_quarantine_path: Path | None = None,
    candidate_quarantine_summary: dict[str, Any] | None = None,
    tokenizer_candidate_summary: dict[str, Any] | None = None,
    tokenizer_manifest_path: Path | None = None,
    tokenizer_report_path: Path | None = None,
    replay_mixture_summary: dict[str, Any] | None = None,
    replay_mixture_path: Path | None = None,
    sweep_plan_summary: dict[str, Any] | None = None,
    sweep_plan_path: Path | None = None,
) -> dict[str, Any]:
    candidate_examples = source_mixture(training_examples)["candidate_examples"]
    candidate_status = "candidate_quarantine_missing"
    if candidate_examples > 0:
        candidate_status = "training_examples_contain_candidates"
    elif candidate_quarantine_summary is not None:
        if candidate_quarantine_summary.get("candidate_count", 0) == 0:
            candidate_status = "candidate_quarantine_empty"
        elif candidate_quarantine_summary.get("not_training_eligible_count", 0) > 0:
            candidate_status = "candidate_quarantine_holds_candidates"
        else:
            candidate_status = "candidate_quarantine_all_candidates_admitted"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "training_plan",
        "component": component,
        "run_id": run_id,
        "allowed_data_sources": [
            str(corpus_dir / "glossary.json"),
            str(corpus_dir / "grammar.json"),
            str(corpus_dir / "admissions.jsonl"),
            str(train_text_path),
            *[str(path) for path in eval_paths],
        ],
        "data_boundary": {
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "unledgered_training_text": False,
        },
        "hygiene_report": str(hygiene_path),
        "eval_sets": eval_set_counts(eval_paths),
        "examples": {
            "base_examples": len(training_examples),
            "scheduled_examples": len(training_pool),
            "base_source_mixture": source_mixture(training_examples),
            "scheduled_source_mixture": source_mixture(training_pool),
            "rare_profile_coverage": rare_profile_coverage(training_examples),
        },
        "candidate_policy": {
            "candidate_examples": candidate_examples,
            "candidate_ratio": (
                candidate_examples / len(training_examples)
                if training_examples
                else 0.0
            ),
            "status": candidate_status,
            "candidate_records_are_training_data": False,
            "rule": "Candidate records are excluded from training until admitted into the ledgered corpus and converted into curriculum lessons.",
            "candidate_quarantine": {
                "path": (
                    str(candidate_quarantine_path)
                    if candidate_quarantine_path is not None
                    else None
                ),
                "summary": candidate_quarantine_summary,
            },
        },
        "tokenizer_candidate": {
            "status": (
                "written" if tokenizer_candidate_summary is not None else "not_planned"
            ),
            "manifest_path": (
                str(tokenizer_manifest_path)
                if tokenizer_manifest_path is not None
                else None
            ),
            "report_path": (
                str(tokenizer_report_path)
                if tokenizer_report_path is not None
                else None
            ),
            "summary": tokenizer_candidate_summary,
            "active_tokenizer_changed": False,
            "rule": "Tokenizer candidate artifacts are evidence only until guarded model-evaluation gates accept a tokenizer promotion.",
        },
        "replay_plan": {
            "status": "planned" if replay_plan_path is not None else "not_applicable",
            "path": str(replay_plan_path) if replay_plan_path is not None else None,
        },
        "replay_mixture": {
            "status": "written" if replay_mixture_summary is not None else "not_planned",
            "path": str(replay_mixture_path) if replay_mixture_path is not None else None,
            "summary": replay_mixture_summary,
            "rule": "Replay mixtures must name new lessons, retained facts, unknown-policy probes, tokenizer stress strings, and heldout/paraphrase evidence.",
        },
        "controlled_sweep": {
            "status": "written" if sweep_plan_summary is not None else "not_planned",
            "path": str(sweep_plan_path) if sweep_plan_path is not None else None,
            "summary": sweep_plan_summary,
            "rule": "Transformer screens must record their tokenizer, architecture, optimizer, and training-budget axes before promotion evidence is trusted.",
        },
        "planned_artifacts": [
            str(path) for path in (planned_artifacts or [])
        ],
    }


def attach_replay_plan_summary(
    training_plan: dict[str, Any],
    replay_plan: dict[str, Any],
    replay_plan_path: Path,
) -> dict[str, Any]:
    updated = dict(training_plan)
    profiles = replay_plan.get("profiles", {})
    updated["replay_plan"] = {
        "status": "written",
        "path": str(replay_plan_path),
        "profile_aware_targets": replay_plan.get("profile_aware_targets"),
        "branch_count": replay_plan.get("branch_count"),
        "replay_count": replay_plan.get("replay_count"),
        "profile_count": len(profiles),
        "profiles_with_missing_targets": [
            profile
            for profile, summary in sorted(profiles.items())
            if summary.get("missing_target_count", 0) > 0
        ],
    }
    return updated
