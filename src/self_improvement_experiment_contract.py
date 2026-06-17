"""Shared self-improvement experiment contract data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


SELF_IMPROVEMENT_COMPONENT = "self-improvement-answer-cycle"
SELF_IMPROVEMENT_RECIPE_ID = "self-improve-answer-cycle:v0.77"
DEFAULT_EXPERIMENT_VERSION = "v0.77"


def experiment_version(args: argparse.Namespace) -> str:
    return getattr(args, "experiment_version", DEFAULT_EXPERIMENT_VERSION)


def experiment_run_id(run_dir: Path, attempt_dir: Path) -> str:
    return attempt_dir.name or run_dir.name


def self_improvement_hypothesis(args: argparse.Namespace) -> str:
    return getattr(args, "experiment_hypothesis", None) or (
        "A closed-world answer-cycle can update answer_model and answer_decoder "
        "weights from admitted corpus lessons while preserving exact evals, "
        "prompt-leakage controls, and forgetting gates."
    )


def self_improvement_notes(args: argparse.Namespace) -> list[str]:
    return getattr(args, "experiment_note", None) or []


def allowed_data_sources(args: argparse.Namespace, train_text_path: Path) -> list[str]:
    return [
        str(args.corpus_dir / "admissions.jsonl"),
        str(args.corpus_dir / "glossary.json"),
        str(args.corpus_dir / "grammar.json"),
        str(train_text_path),
    ]


def planned_experiment_artifacts(run_dir: Path, attempt_dir: Path) -> list[str]:
    artifact_names = [
        "corpus_hygiene.json",
        "training_plan.json",
        "training_recipe.json",
        "candidate_quarantine.json",
        "closed_world_verifier.json",
        "constraint_first_promotion.json",
        "experiment_intent.json",
        "self_improvement_report.json",
    ]
    return [
        str(attempt_dir / "answer" / "answer_model.json"),
        str(attempt_dir / "decoder" / "answer_decoder.json"),
        str(attempt_dir / "corpus_snapshot.json"),
        str(attempt_dir / "corpus_diff.json"),
        *[str(attempt_dir / name) for name in artifact_names],
        *[str(run_dir / name) for name in artifact_names],
    ]


def self_improvement_acceptance_gates() -> list[dict[str, Any]]:
    return [
        {
            "name": "training_recipe",
            "rule": "A recipe artifact must bind model, data, objective, optimizer, artifacts, and gates.",
            "required": True,
        },
        {
            "name": "closed_world_verifier",
            "rule": "Training plan, hygiene, and candidate quarantine checks must pass before training.",
            "required": True,
        },
        {
            "name": "constraint_first_promotion",
            "rule": "Quality metrics can influence promotion only after closed-world constraints pass.",
            "required": True,
        },
        {
            "name": "admission_probe_audit",
            "rule": "Generated probes for every admitted fact must pass.",
            "required": True,
        },
        {
            "name": "glossary_probe_audit",
            "rule": "Glossary-derived probes must remain exact.",
            "required": True,
        },
        {
            "name": "heldout_prompt_leakage",
            "rule": "Heldout prompts must not appear in training lessons.",
            "required": True,
        },
        {
            "name": "owner_heldout_prompt_leakage",
            "rule": "Protected owner heldout prompts must not appear in lessons.",
            "required": True,
        },
        {
            "name": "forgetting_audit",
            "rule": "Current evals may not regress against the comparison report.",
            "required": True,
        },
        {
            "name": "exact_eval_audit",
            "rule": "Responder, answer model, and decoder evals must be exact.",
            "required": True,
        },
        {
            "name": "promotion_gate",
            "rule": "All required audits must pass before the result can promote.",
            "required": True,
        },
    ]


def self_improvement_failure_criteria() -> list[str]:
    return [
        "Any required probe, leakage, forgetting, or exact-eval audit fails.",
        "Training writes checkpoints without a matching report and intent artifact.",
        "The run uses pretrained weights, pretrained tokenizers, or external embeddings.",
        "The deterministic closed-world verifier rejects the training plan.",
    ]
