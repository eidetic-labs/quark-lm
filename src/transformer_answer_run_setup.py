"""Preparation workflow for transformer answer-training runs."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answer_model import (
    DEFAULT_EVALS as DEFAULT_ANSWER_EVALS,
    AnswerExample,
    answer_training_pool,
    load_training_examples,
    write_lessons,
)
from tokenizer_protocol import TokenizerProtocol
from probes import read_jsonl
from memory_retrieval import (
    build_retrieval_memory_report,
    write_retrieval_memory_report,
)
from transformer_answer_governance_setup import (
    prepare_transformer_answer_governance,
)
from transformer_direct_answer_evaluation import audit_prompt_context_coverage
from transformer_experiment import (
    TransformerRunArtifacts,
    direct_answer_is_profile_aware,
)
from transformer_model import (
    GenerationConfig,
    generation_config_from_args,
    optimization_config_from_args,
)
from transformer_optimizer import ScalarOptimizer, load_optimizer_state
from transformer_text_commands import ensure_curriculum
from transformer_training import JsonlHistoryWriter
from transformer_training_tokenizer import training_tokenizer


@dataclass
class TransformerAnswerRunSetup:
    tokenizer: TokenizerProtocol
    examples: list[AnswerExample]
    training_pool: list[AnswerExample]
    model: Any
    resume_metadata: dict[str, Any]
    optimizer: ScalarOptimizer
    generation_config: GenerationConfig
    rng: random.Random
    artifacts: TransformerRunArtifacts
    experiment_path: Path
    experiment_intent: dict[str, Any]
    hygiene_path: Path
    training_plan_path: Path
    training_recipe_path: Path
    candidate_quarantine_path: Path
    verifier_path: Path
    constraint_first_path: Path
    retrieval_memory_path: Path
    memory_consolidation_plan_path: Path
    replay_mixture_path: Path
    sweep_plan_path: Path
    candidate_quarantine: dict[str, Any]
    training_recipe: dict[str, Any]
    corpus_hygiene: dict[str, Any]
    training_plan: dict[str, Any]
    replay_mixture: dict[str, Any]
    sweep_plan: dict[str, Any]
    closed_world_verifier: dict[str, Any]
    history_path: Path
    lessons_path: Path
    history_writer: JsonlHistoryWriter
    eval_records: dict[str, list[dict[str, Any]]]
    retrieval_memory: dict[str, Any]
    context_coverage: dict[str, Any]
    candidates: list[str]
    training_candidates: list[str]
    eval_candidates: dict[str, list[str]]


def prepare_transformer_answer_run(
    args: argparse.Namespace,
    initialize_transformer_for_training_fn: Callable[
        [argparse.Namespace, TokenizerProtocol],
        tuple[Any, dict[str, Any]],
    ],
    model_class: type[Any],
) -> TransformerAnswerRunSetup:
    ensure_curriculum(args.train_text, args.valid)
    train_text = args.train_text.read_text(encoding="utf-8")
    tokenizer = training_tokenizer(args, train_text, model_class)
    examples = load_training_examples(args.train_text, args.corpus_dir)
    training_pool = answer_training_pool(examples)
    model, resume_metadata = initialize_transformer_for_training_fn(args, tokenizer)
    optimizer = load_optimizer_state(
        args.resume_optimizer,
        optimization_config_from_args(args),
    )
    model.active_optimizer = optimizer
    generation_config = generation_config_from_args(args)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    artifacts = TransformerRunArtifacts.from_run(
        args.run,
        direct_profile_aware=direct_answer_is_profile_aware(args),
    )
    retrieval_memory_path = artifacts.retrieval_memory
    governance = prepare_transformer_answer_governance(
        args=args,
        tokenizer=tokenizer,
        examples=examples,
        training_pool=training_pool,
        artifacts=artifacts,
    )
    history_path = artifacts.metrics_history
    lessons_path = artifacts.lessons
    write_lessons(examples, lessons_path)
    history_writer = JsonlHistoryWriter(history_path)
    eval_records = {path.stem: read_jsonl(path) for path in DEFAULT_ANSWER_EVALS}
    retrieval_memory = build_retrieval_memory_report(
        args.corpus_dir,
        DEFAULT_ANSWER_EVALS,
    )
    write_retrieval_memory_report(retrieval_memory_path, retrieval_memory)
    context_coverage = {
        name: audit_prompt_context_coverage(records, args.context_size)
        for name, records in sorted(eval_records.items())
    }
    candidates = sorted(
        {
            record["target"]
            for records in eval_records.values()
            for record in records
        }
    )
    training_candidates = sorted(
        {example.target for example in examples}
        | {
            record["target"]
            for records in eval_records.values()
            for record in records
        }
    )
    eval_candidates = {
        name: sorted({record["target"] for record in records})
        for name, records in eval_records.items()
    }
    return TransformerAnswerRunSetup(
        tokenizer=tokenizer,
        examples=examples,
        training_pool=training_pool,
        model=model,
        resume_metadata=resume_metadata,
        optimizer=optimizer,
        generation_config=generation_config,
        rng=rng,
        artifacts=artifacts,
        experiment_path=governance.experiment_path,
        experiment_intent=governance.experiment_intent,
        hygiene_path=governance.hygiene_path,
        training_plan_path=governance.training_plan_path,
        training_recipe_path=governance.training_recipe_path,
        candidate_quarantine_path=governance.candidate_quarantine_path,
        verifier_path=governance.verifier_path,
        constraint_first_path=governance.constraint_first_path,
        retrieval_memory_path=retrieval_memory_path,
        memory_consolidation_plan_path=governance.memory_consolidation_plan_path,
        replay_mixture_path=governance.replay_mixture_path,
        sweep_plan_path=governance.sweep_plan_path,
        candidate_quarantine=governance.candidate_quarantine,
        training_recipe=governance.training_recipe,
        corpus_hygiene=governance.corpus_hygiene,
        training_plan=governance.training_plan,
        replay_mixture=governance.replay_mixture,
        sweep_plan=governance.sweep_plan,
        closed_world_verifier=governance.closed_world_verifier,
        history_path=history_path,
        lessons_path=lessons_path,
        history_writer=history_writer,
        eval_records=eval_records,
        retrieval_memory=retrieval_memory,
        context_coverage=context_coverage,
        candidates=candidates,
        training_candidates=training_candidates,
        eval_candidates=eval_candidates,
    )
