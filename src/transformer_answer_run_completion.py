"""Final completion of transformer answer-training runs."""

from __future__ import annotations

from typing import Any, Callable

from experiment_registry import record_experiment_decision, write_experiment_intent
from transformer_answer_auxiliary_stages import (
    train_answer_generator_stage,
    train_answer_selector_stage,
)
from transformer_answer_run_finalization import finalize_transformer_answer_run


def complete_transformer_answer_training_run(
    *,
    args: Any,
    setup: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    training_plan: dict[str, Any],
    baseline: dict[str, Any],
    last_snapshot: dict[str, Any],
    post_direct_candidate_snapshot: dict[str, Any] | None,
    post_direct_candidate_snapshot_skipped: bool,
    direct_answer_metrics: dict[str, Any] | None,
    train_selector: Callable[..., dict[str, Any]] = train_answer_selector_stage,
    train_generator: Callable[..., dict[str, Any]] = train_answer_generator_stage,
    finalize_run: Callable[..., dict[str, Any]] = finalize_transformer_answer_run,
) -> dict[str, Any]:
    selector_metrics = train_selector(
        args,
        model,
        tokenizer,
        setup.examples,
        setup.training_pool,
        setup.eval_records,
        setup.eval_candidates,
        setup.candidates,
    )
    generator_metrics = train_generator(
        args,
        model,
        tokenizer,
        setup.examples,
        setup.eval_records,
    )
    return finalize_run(
        args,
        model,
        tokenizer,
        optimizer,
        setup.artifacts,
        setup.resume_metadata,
        setup.experiment_intent,
        setup.history_path,
        setup.lessons_path,
        setup.examples,
        setup.training_pool,
        setup.candidates,
        setup.training_candidates,
        setup.generation_config,
        setup.context_coverage,
        baseline,
        last_snapshot,
        post_direct_candidate_snapshot,
        post_direct_candidate_snapshot_skipped,
        direct_answer_metrics,
        selector_metrics,
        generator_metrics,
        setup.corpus_hygiene,
        setup.hygiene_path,
        setup.retrieval_memory,
        setup.retrieval_memory_path,
        setup.memory_consolidation_plan_path,
        training_plan,
        setup.training_plan_path,
        setup.training_recipe,
        setup.training_recipe_path,
        setup.candidate_quarantine,
        setup.candidate_quarantine_path,
        setup.closed_world_verifier,
        setup.verifier_path,
        setup.constraint_first_path,
        setup.experiment_path,
        record_experiment_decision,
        write_experiment_intent,
    )
