"""Closed-world governance artifact setup for transformer answer runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from answer_model import DEFAULT_EVALS as DEFAULT_ANSWER_EVALS, AnswerExample
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
from experiment_registry import write_experiment_intent
from probes import read_jsonl
from tokenizer_protocol import TokenizerProtocol
from training_recipe_core import attach_recipe_summary, write_training_recipe
from transformer_experiment import (
    TransformerRunArtifacts,
    transformer_experiment_intent,
    transformer_training_recipe_id,
)
from transformer_replay_mixture import (
    build_transformer_replay_mixture_report,
    replay_mixture_summary,
    write_transformer_replay_mixture_report,
)
from transformer_sweep_plan import (
    build_transformer_sweep_plan,
    sweep_plan_summary,
    write_transformer_sweep_plan,
)
from transformer_text_commands import transformer_training_recipe


@dataclass(frozen=True)
class TransformerAnswerGovernanceSetup:
    experiment_path: Path
    experiment_intent: dict[str, object]
    hygiene_path: Path
    training_plan_path: Path
    training_recipe_path: Path
    candidate_quarantine_path: Path
    verifier_path: Path
    constraint_first_path: Path
    memory_consolidation_plan_path: Path
    replay_mixture_path: Path
    sweep_plan_path: Path
    candidate_quarantine: dict[str, object]
    training_recipe: dict[str, object]
    corpus_hygiene: dict[str, object]
    training_plan: dict[str, object]
    replay_mixture: dict[str, object]
    sweep_plan: dict[str, object]
    closed_world_verifier: dict[str, object]


def prepare_transformer_answer_governance(
    *,
    args: argparse.Namespace,
    tokenizer: TokenizerProtocol,
    examples: list[AnswerExample],
    training_pool: list[AnswerExample],
    artifacts: TransformerRunArtifacts,
) -> TransformerAnswerGovernanceSetup:
    experiment_path = artifacts.experiment_intent
    experiment_intent = transformer_experiment_intent(args)
    write_experiment_intent(experiment_path, experiment_intent)

    candidate_quarantine = build_candidate_quarantine_manifest(
        "transformer-answer-train",
        args.run.name,
    )
    write_candidate_quarantine(artifacts.candidate_quarantine, candidate_quarantine)
    candidate_summary = candidate_quarantine_summary(candidate_quarantine)
    planned_artifacts = artifacts.training_plan_artifacts()
    eval_records = {path.stem: read_jsonl(path) for path in DEFAULT_ANSWER_EVALS}
    replay_mixture = build_transformer_replay_mixture_report(
        run_id=args.run.name,
        train_text_path=args.train_text,
        examples=examples,
        training_pool=training_pool,
        eval_records=eval_records,
        admissions=read_jsonl(args.corpus_dir / "admissions.jsonl"),
    )
    write_transformer_replay_mixture_report(
        artifacts.replay_mixture_report,
        replay_mixture,
    )
    sweep_plan = build_transformer_sweep_plan(
        args,
        tokenizer,
        recipe_id=transformer_training_recipe_id(args),
    )
    write_transformer_sweep_plan(artifacts.sweep_plan, sweep_plan)
    training_recipe = transformer_training_recipe(
        args,
        tokenizer,
        planned_artifacts,
        experiment_intent["acceptance_gates"],
        artifacts.replay_plan,
        artifacts.replay_mixture_report,
        replay_mixture_summary(replay_mixture),
        artifacts.sweep_plan,
        sweep_plan_summary(sweep_plan),
    )
    write_training_recipe(artifacts.training_recipe, training_recipe)

    corpus_hygiene = build_corpus_hygiene_report(
        "transformer-answer-train",
        args.corpus_dir,
        args.train_text,
        DEFAULT_ANSWER_EVALS,
        examples,
    )
    training_plan = build_training_plan(
        "transformer-answer-train",
        args.run.name,
        args.train_text,
        args.corpus_dir,
        DEFAULT_ANSWER_EVALS,
        examples,
        training_pool,
        artifacts.corpus_hygiene,
        planned_artifacts=planned_artifacts,
        replay_plan_path=artifacts.replay_plan,
        candidate_quarantine_path=artifacts.candidate_quarantine,
        candidate_quarantine_summary=candidate_summary,
        replay_mixture_summary=replay_mixture_summary(replay_mixture),
        replay_mixture_path=artifacts.replay_mixture_report,
        sweep_plan_summary=sweep_plan_summary(sweep_plan),
        sweep_plan_path=artifacts.sweep_plan,
    )
    training_plan = attach_recipe_summary(
        training_plan,
        training_recipe,
        artifacts.training_recipe,
    )
    write_json_artifact(artifacts.corpus_hygiene, corpus_hygiene)
    write_json_artifact(artifacts.training_plan, training_plan)

    closed_world_verifier = verify_training_plan(
        training_plan,
        corpus_hygiene=corpus_hygiene,
        candidate_quarantine=candidate_quarantine,
        subject_path=artifacts.training_plan,
        verifier_path=artifacts.closed_world_verifier,
    )
    write_verifier_report(artifacts.closed_world_verifier, closed_world_verifier)
    training_plan = attach_verifier_summary(
        training_plan,
        closed_world_verifier,
        artifacts.closed_world_verifier,
    )
    write_json_artifact(artifacts.training_plan, training_plan)
    if not closed_world_verifier["passed"]:
        raise ValueError("closed-world verifier rejected the training plan")

    return TransformerAnswerGovernanceSetup(
        experiment_path=experiment_path,
        experiment_intent=experiment_intent,
        hygiene_path=artifacts.corpus_hygiene,
        training_plan_path=artifacts.training_plan,
        training_recipe_path=artifacts.training_recipe,
        candidate_quarantine_path=artifacts.candidate_quarantine,
        verifier_path=artifacts.closed_world_verifier,
        constraint_first_path=artifacts.constraint_first_promotion,
        memory_consolidation_plan_path=artifacts.memory_consolidation_plan,
        replay_mixture_path=artifacts.replay_mixture_report,
        sweep_plan_path=artifacts.sweep_plan,
        candidate_quarantine=candidate_quarantine,
        training_recipe=training_recipe,
        corpus_hygiene=corpus_hygiene,
        training_plan=training_plan,
        replay_mixture=replay_mixture,
        sweep_plan=sweep_plan,
        closed_world_verifier=closed_world_verifier,
    )
