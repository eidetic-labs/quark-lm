"""Direct-answer training metrics assembly."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from branch_diversity_snapshot_coverage import branch_diversity_snapshot_target_coverage_by_profile
from transformer_direct_answer_metric_sections import (
    build_baseline_floor_metric_section,
    build_memory_consolidation_metric_section,
)
from transformer_direct_answer_frontier_reference import (
    build_direct_answer_frontier_reference,
)
from transformer_experiment import TRAINING_DATA_DESCRIPTION
from transformer_model import GenerationConfig


def build_direct_answer_metrics(
    *,
    args: argparse.Namespace,
    direct_history_path: Path,
    direct_steps_to_run: int,
    direct_training_example_count: int,
    direct_profile_aware_targets: bool,
    direct_replay_plan_path: Path | None,
    direct_replay_plan: dict[str, Any] | None,
    direct_replay_prediction_overrides: Mapping[Any, Any] | None,
    direct_replay_prediction_anchors_active: bool,
    direct_memory_consolidation_source_plan_path: Path | None,
    direct_memory_consolidation_target_profiles: list[str],
    direct_memory_consolidation_top_priority_profiles: list[str],
    direct_memory_consolidation_collapsed_memory_backed_profiles: list[str],
    direct_memory_consolidation_missing_first_token_values: dict[str, list[str]],
    direct_memory_consolidation_missing_first_token_ids: dict[str, list[int]],
    direct_memory_consolidation_profile_specific_missing_first_token_target_map: dict[
        str, list[str]
    ],
    direct_answer_update_guard: dict[str, Any],
    direct_answer_restored_best_branch_snapshot: bool,
    best_direct_snapshot_step: int,
    best_direct_snapshot_score: tuple[Any, ...],
    direct_baseline: dict[str, Any],
    direct_answer_training_skipped: bool,
    direct_answer_skip_reason: str | None,
    branch_context_gate: dict[str, Any],
    post_direct_candidate_snapshot_skipped: bool,
    generation_config: GenerationConfig,
    direct_answer_terminator: str,
    context_coverage: dict[str, Any],
    last_direct_snapshot: dict[str, Any],
    routing_repair_batch_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_floor_metrics = build_baseline_floor_metric_section(
        direct_answer_update_guard,
    )
    return {
        "architecture": "tiny-decoder-only-transformer-direct-answer",
        "checkpoint": str(args.run / "transformer_answer.json"),
        "history": str(direct_history_path),
        "steps": args.direct_answer_steps,
        "actual_steps": direct_steps_to_run,
        "training_examples": direct_training_example_count,
        "learning_rate": args.direct_answer_learning_rate,
        "direct_answer_eval_every": args.direct_answer_eval_every,
        "direct_answer_snapshot_mode": args.direct_answer_snapshot_mode,
        "direct_answer_evals_skipped": (
            args.direct_answer_snapshot_mode == "branch-only"
        ),
        "direct_answer_mode": args.direct_answer_mode,
        "direct_answer_profile_aware_targets": direct_profile_aware_targets,
        "direct_answer_replay_plan": (
            str(direct_replay_plan_path)
            if direct_replay_plan_path is not None
            else None
        ),
        "direct_answer_replay_plan_summary": direct_replay_plan,
        "direct_answer_replay_prediction_anchor_count": (
            len(direct_replay_prediction_overrides)
            if direct_replay_prediction_overrides is not None
            else 0
        ),
        "direct_answer_replay_prediction_anchors_active": (
            direct_replay_prediction_anchors_active
        ),
        **baseline_floor_metrics.fields,
        **build_memory_consolidation_metric_section(
            source_plan_path=direct_memory_consolidation_source_plan_path,
            target_profiles=direct_memory_consolidation_target_profiles,
            top_priority_profiles=direct_memory_consolidation_top_priority_profiles,
            collapsed_memory_backed_profiles=(
                direct_memory_consolidation_collapsed_memory_backed_profiles
            ),
            missing_first_token_values=(
                direct_memory_consolidation_missing_first_token_values
            ),
            missing_first_token_ids=direct_memory_consolidation_missing_first_token_ids,
            profile_specific_missing_first_token_target_map=(
                direct_memory_consolidation_profile_specific_missing_first_token_target_map
            ),
            remaining_collapsed_memory_active=(
                baseline_floor_metrics.remaining_collapsed_memory_active
            ),
            profile_specific_missing_first_token_active=(
                baseline_floor_metrics.profile_specific_missing_first_token_active
            ),
        ),
        "direct_answer_update_guard": direct_answer_update_guard,
        "direct_answer_negative_weight": args.direct_answer_negative_weight,
        "direct_answer_positive_weight": args.direct_answer_positive_weight,
        "direct_answer_contrast_weight": args.direct_answer_contrast_weight,
        "direct_answer_recovery_steps": args.direct_answer_recovery_steps,
        "direct_answer_branch_position": args.direct_answer_branch_position,
        "direct_answer_branch_span": args.direct_answer_branch_span,
        "direct_answer_branch_batch_size": args.direct_answer_branch_batch_size,
        "direct_answer_hard_negatives": args.direct_answer_hard_negatives,
        "direct_answer_train_top_layer_only": args.direct_answer_train_top_layer_only,
        "direct_answer_freeze_output_bias": args.direct_answer_freeze_output_bias,
        "direct_answer_restore_best_branch_snapshot": (
            args.direct_answer_restore_best_branch_snapshot
        ),
        "direct_answer_restored_best_branch_snapshot": (
            direct_answer_restored_best_branch_snapshot
        ),
        "direct_answer_best_branch_snapshot_step": best_direct_snapshot_step,
        "direct_answer_best_branch_snapshot_score": list(best_direct_snapshot_score),
        "direct_answer_branch_snapshot_coverage_floor": (
            branch_diversity_snapshot_target_coverage_by_profile(direct_baseline)
        ),
        "direct_answer_frontier_reference": build_direct_answer_frontier_reference(
            args=args,
            direct_baseline=direct_baseline,
            final_snapshot=last_direct_snapshot,
        ),
        "routing_repair_batch_evidence": routing_repair_batch_evidence,
        "direct_answer_require_branch_context_gate": (
            args.direct_answer_require_branch_context_gate
        ),
        "direct_answer_training_skipped": direct_answer_training_skipped,
        "direct_answer_skip_reason": direct_answer_skip_reason,
        "direct_answer_branch_context_gate": branch_context_gate,
        "post_direct_candidate_snapshot_skipped": post_direct_candidate_snapshot_skipped,
        "direct_answer_sequence_interval": args.direct_answer_sequence_interval,
        "direct_answer_rollout_interval": args.direct_answer_rollout_interval,
        "max_new_chars": args.direct_answer_max_new_chars,
        "generation_config": asdict(generation_config),
        "terminator": repr(direct_answer_terminator),
        "context_coverage": context_coverage,
        "baseline": direct_baseline,
        "final": last_direct_snapshot,
        "uses_answer_candidates": False,
        "auxiliary_weights": False,
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "training_data": TRAINING_DATA_DESCRIPTION,
    }
