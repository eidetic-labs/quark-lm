"""Final artifact writing for transformer answer-training runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memory_consolidation import (
    build_memory_consolidation_plan,
    write_memory_consolidation_plan,
)
from constraint_first_report import write_constraint_first_report
from transformer_constraints import transformer_constraint_report
from transformer_experiment import (
    TRAINING_DATA_DESCRIPTION,
    transformer_experiment_decision,
)
from transformer_model import (
    TRANSFORMER_ARCHITECTURE,
    TRANSFORMER_CHECKPOINT_FORMAT,
    GenerationConfig,
    transformer_run_metadata,
)
from transformer_optimizer import ScalarOptimizer, save_optimizer_state


def finalize_transformer_answer_run(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    optimizer: ScalarOptimizer,
    artifacts: Any,
    resume_metadata: dict[str, Any],
    experiment_intent: dict[str, Any],
    history_path: Path,
    lessons_path: Path,
    examples: list[Any],
    training_pool: list[Any],
    candidates: list[str],
    training_candidates: list[str],
    generation_config: GenerationConfig,
    context_coverage: dict[str, Any],
    baseline: dict[str, Any],
    last_snapshot: dict[str, Any],
    post_direct_candidate_snapshot: dict[str, Any] | None,
    post_direct_candidate_snapshot_skipped: bool,
    direct_answer_metrics: dict[str, Any] | None,
    selector_metrics: dict[str, Any] | None,
    generator_metrics: dict[str, Any] | None,
    corpus_hygiene: dict[str, Any],
    hygiene_path: Path,
    retrieval_memory: dict[str, Any],
    retrieval_memory_path: Path,
    memory_consolidation_plan_path: Path,
    replay_mixture: dict[str, Any],
    replay_mixture_path: Path,
    sweep_plan: dict[str, Any],
    sweep_plan_path: Path,
    training_plan: dict[str, Any],
    training_plan_path: Path,
    training_recipe: dict[str, Any],
    training_recipe_path: Path,
    candidate_quarantine: dict[str, Any],
    candidate_quarantine_path: Path,
    closed_world_verifier: dict[str, Any],
    verifier_path: Path,
    constraint_first_path: Path,
    experiment_path: Path,
    record_experiment_decision_fn: Any,
    write_experiment_intent_fn: Any,
) -> dict[str, Any]:
    checkpoint_path = artifacts.checkpoint
    optimizer_path = artifacts.optimizer_state
    save_optimizer_state(optimizer_path, optimizer)
    checkpoint_metadata = transformer_run_metadata(
        args,
        tokenizer,
        optimizer,
        "answer-train",
        resume_metadata,
    )
    model.save(checkpoint_path, tokenizer, checkpoint_metadata)
    tokenizer.save(artifacts.tokenizer)
    metrics = {
        "architecture": TRANSFORMER_ARCHITECTURE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": TRANSFORMER_CHECKPOINT_FORMAT,
        "optimizer_state": str(optimizer_path),
        "optimizer": optimizer.summary(),
        "resume": resume_metadata,
        "history": str(history_path),
        "lessons": str(lessons_path),
        "steps": args.steps,
        "examples": len(examples),
        "training_examples": len(training_pool),
        "candidate_count": len(candidates),
        "training_candidate_count": len(training_candidates),
        "candidate_scope": args.candidate_scope,
        "include_completions": args.include_completions,
        "generation_config": asdict(generation_config),
        "target_loss_weight": args.target_loss_weight,
        "choice_loss_weight": args.choice_loss_weight,
        "choice_negatives": args.choice_negatives,
        "choice_max_chars": args.choice_max_chars,
        "vocab_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "num_layers": args.num_layers,
        "attention_heads": args.attention_heads,
        "use_layer_norm": args.use_layer_norm,
        "use_pre_layer_norm": args.use_pre_layer_norm,
        "use_rms_norm": args.use_rms_norm,
        "layer_norm_epsilon": args.layer_norm_epsilon,
        "use_gated_mlp": args.use_gated_mlp,
        "tie_output_embeddings": args.tie_output_embeddings,
        "use_rotary_positions": args.use_rotary_positions,
        "use_kv_cache_path": args.use_kv_cache_path,
        "use_context_mean": args.use_context_mean,
        "use_context_projection": args.use_context_projection,
        "use_prompt_prefix_projection": args.use_prompt_prefix_projection,
        "use_prompt_position_projection": args.use_prompt_position_projection,
        "prompt_position_projection_scale": args.prompt_position_projection_scale,
        "use_prompt_attention_summary": args.use_prompt_attention_summary,
        "context_coverage": context_coverage,
        "baseline": baseline,
        "final": last_snapshot,
        "post_direct_candidate_snapshot": post_direct_candidate_snapshot,
        "post_direct_candidate_snapshot_skipped": post_direct_candidate_snapshot_skipped,
        "direct_answer": direct_answer_metrics,
        "answer_selector": selector_metrics,
        "answer_generator": generator_metrics,
        "corpus_hygiene": corpus_hygiene,
        "corpus_hygiene_path": str(hygiene_path),
        "retrieval_memory": {
            "path": str(retrieval_memory_path),
            "summary": retrieval_memory["summary"],
            "memory": retrieval_memory["memory"],
            "dataset_exclusivity": retrieval_memory["dataset_exclusivity"],
            "self_improvement": retrieval_memory["self_improvement"],
        },
        "memory_consolidation_plan_path": str(memory_consolidation_plan_path),
        "replay_mixture_report": replay_mixture,
        "replay_mixture_report_path": str(replay_mixture_path),
        "sweep_plan": sweep_plan,
        "sweep_plan_path": str(sweep_plan_path),
        "training_plan": training_plan,
        "training_plan_path": str(training_plan_path),
        "training_recipe": training_recipe,
        "training_recipe_path": str(training_recipe_path),
        "candidate_quarantine": candidate_quarantine,
        "candidate_quarantine_path": str(candidate_quarantine_path),
        "closed_world_verifier": closed_world_verifier,
        "closed_world_verifier_path": str(verifier_path),
        "constraint_first_promotion_path": str(constraint_first_path),
        "experiment_intent_path": str(experiment_path),
        "metrics_path": str(artifacts.metrics),
        "run_id": args.run.name,
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "tokenizer": checkpoint_metadata["dataset"]["tokenizer"],
        "tokenizer_type": checkpoint_metadata["dataset"]["tokenizer_type"],
        "tokenizer_manifest_hash": checkpoint_metadata["dataset"][
            "tokenizer_manifest_hash"
        ],
        "training_data": TRAINING_DATA_DESCRIPTION,
    }
    memory_consolidation_plan = build_memory_consolidation_plan(
        retrieval_memory,
        metrics,
    )
    write_memory_consolidation_plan(
        memory_consolidation_plan_path,
        memory_consolidation_plan,
    )
    metrics["memory_consolidation_plan"] = {
        "path": str(memory_consolidation_plan_path),
        "summary": memory_consolidation_plan["summary"],
        "dataset_exclusivity": memory_consolidation_plan["dataset_exclusivity"],
        "self_improvement": memory_consolidation_plan["self_improvement"],
    }
    metrics["constraint_first_promotion"] = transformer_constraint_report(metrics)
    write_constraint_first_report(
        constraint_first_path,
        metrics["constraint_first_promotion"],
    )
    status, summary, evidence = transformer_experiment_decision(metrics)
    metrics["experiment_intent"] = record_experiment_decision_fn(
        experiment_intent,
        status,
        summary,
        evidence,
    )
    write_experiment_intent_fn(experiment_path, metrics["experiment_intent"])
    with artifacts.metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics
