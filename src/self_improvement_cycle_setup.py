"""Setup and verification for self-improvement answer cycles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answer_model import DEFAULT_EVALS, answer_training_pool, load_training_examples
from candidate_quarantine import (
    build_candidate_quarantine_manifest,
    candidate_quarantine_summary,
    write_candidate_quarantine,
)
from closed_world_training_plan_verifier import verify_training_plan
from closed_world_verifier_reports import (
    attach_verifier_summary,
    write_verifier_report,
)
from corpus_hygiene import (
    build_corpus_hygiene_report,
    build_training_plan,
    write_json_artifact,
)
from curriculum import build_curriculum, write_curriculum
from experiment_registry import write_experiment_intent
from provenance import corpus_snapshot
from self_improvement_artifacts import next_attempt
from self_improvement_experiment import (
    self_improvement_experiment_intent,
    self_improvement_training_recipe,
)
from self_improvement_tokenizer import build_self_improvement_tokenizer_candidate
from training_recipe_core import attach_recipe_summary, write_training_recipe


@dataclass(frozen=True)
class AnswerCycleSetup:
    curriculum: Any
    snapshot: dict[str, Any]
    train_text_path: Path
    training_examples: list[dict[str, Any]]
    run_dir: Path
    attempt_number: int
    attempt_dir: Path
    answer_run: Path
    decoder_run: Path
    experiment_intent: dict[str, Any]
    corpus_hygiene: dict[str, Any]
    training_plan: dict[str, Any]
    training_recipe: dict[str, Any]
    candidate_quarantine: dict[str, Any]
    tokenizer_candidate: dict[str, Any]
    closed_world_verifier: dict[str, Any]
    constraint_first_path: Path


def prepare_answer_cycle_setup(args: argparse.Namespace) -> AnswerCycleSetup:
    curriculum = build_curriculum(args.corpus_dir, args.seed)
    write_curriculum(curriculum, args.build_dir)
    snapshot = corpus_snapshot(args.corpus_dir)
    train_text_path = args.build_dir / "train.txt"
    training_examples = load_training_examples(train_text_path, args.corpus_dir)
    scheduled_training_examples = answer_training_pool(training_examples)
    run_dir = args.run
    attempt_number, attempt_dir = next_attempt(run_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    answer_run = attempt_dir / "answer"
    decoder_run = attempt_dir / "decoder"
    experiment_intent = self_improvement_experiment_intent(
        args,
        run_dir,
        attempt_dir,
        train_text_path,
    )
    write_experiment_intent(attempt_dir / "experiment_intent.json", experiment_intent)

    hygiene_path = attempt_dir / "corpus_hygiene.json"
    training_plan_path = attempt_dir / "training_plan.json"
    training_recipe_path = attempt_dir / "training_recipe.json"
    candidate_quarantine_path = attempt_dir / "candidate_quarantine.json"
    tokenizer_manifest_path = attempt_dir / "tokenizer_manifest.json"
    tokenizer_report_path = attempt_dir / "tokenizer_report.json"
    verifier_path = attempt_dir / "closed_world_verifier.json"
    constraint_first_path = attempt_dir / "constraint_first_promotion.json"
    candidate_quarantine = build_candidate_quarantine_manifest(
        "self-improvement-answer-cycle",
        attempt_dir.name,
    )
    write_candidate_quarantine(candidate_quarantine_path, candidate_quarantine)
    candidate_summary = candidate_quarantine_summary(candidate_quarantine)
    corpus_hygiene = build_corpus_hygiene_report(
        "self-improvement-answer-cycle",
        args.corpus_dir,
        train_text_path,
        DEFAULT_EVALS,
        training_examples,
    )
    tokenizer_candidate = build_self_improvement_tokenizer_candidate(
        train_text_path,
        training_examples,
        tokenizer_manifest_path,
        tokenizer_report_path,
    )
    planned_artifacts = [
        answer_run / "answer_model.json",
        decoder_run / "answer_decoder.json",
        hygiene_path,
        training_plan_path,
        training_recipe_path,
        candidate_quarantine_path,
        tokenizer_manifest_path,
        tokenizer_report_path,
        verifier_path,
        constraint_first_path,
        attempt_dir / "self_improvement_report.json",
    ]
    training_recipe = self_improvement_training_recipe(
        args,
        run_dir,
        attempt_dir,
        train_text_path,
        planned_artifacts,
        experiment_intent["acceptance_gates"],
        tokenizer_candidate["summary"],
        tokenizer_manifest_path,
        tokenizer_report_path,
    )
    write_training_recipe(training_recipe_path, training_recipe)
    training_plan = build_training_plan(
        "self-improvement-answer-cycle",
        attempt_dir.name,
        train_text_path,
        args.corpus_dir,
        DEFAULT_EVALS,
        training_examples,
        scheduled_training_examples,
        hygiene_path,
        planned_artifacts=planned_artifacts,
        candidate_quarantine_path=candidate_quarantine_path,
        candidate_quarantine_summary=candidate_summary,
        tokenizer_candidate_summary=tokenizer_candidate["summary"],
        tokenizer_manifest_path=tokenizer_manifest_path,
        tokenizer_report_path=tokenizer_report_path,
    )
    training_plan = attach_recipe_summary(
        training_plan,
        training_recipe,
        training_recipe_path,
    )
    write_json_artifact(hygiene_path, corpus_hygiene)
    write_json_artifact(training_plan_path, training_plan)
    closed_world_verifier = verify_training_plan(
        training_plan,
        corpus_hygiene=corpus_hygiene,
        candidate_quarantine=candidate_quarantine,
        subject_path=training_plan_path,
        verifier_path=verifier_path,
    )
    write_verifier_report(verifier_path, closed_world_verifier)
    training_plan = attach_verifier_summary(
        training_plan,
        closed_world_verifier,
        verifier_path,
    )
    write_json_artifact(training_plan_path, training_plan)
    if not closed_world_verifier["passed"]:
        raise ValueError("closed-world verifier rejected the training plan")

    return AnswerCycleSetup(
        curriculum=curriculum,
        snapshot=snapshot,
        train_text_path=train_text_path,
        training_examples=training_examples,
        run_dir=run_dir,
        attempt_number=attempt_number,
        attempt_dir=attempt_dir,
        answer_run=answer_run,
        decoder_run=decoder_run,
        experiment_intent=experiment_intent,
        corpus_hygiene=corpus_hygiene,
        training_plan=training_plan,
        training_recipe=training_recipe,
        candidate_quarantine=candidate_quarantine,
        tokenizer_candidate=tokenizer_candidate,
        closed_world_verifier=closed_world_verifier,
        constraint_first_path=constraint_first_path,
    )
