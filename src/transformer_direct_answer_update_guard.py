"""Direct-answer update-guard accounting helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from autograd import Scalar
from branch_diversity_snapshots import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_by_profile,
    branch_diversity_snapshot_target_coverage_diagnostics,
)
from tokenizer import CharTokenizer
from transformer_math import exclude_scalars
from transformer_optimizer import ScalarOptimizer

STABILIZATION_UPDATE_SHAPES = {
    "stabilization",
    "profile_targeted_stabilization",
    "sequential_profile_stabilization",
    "calibrated_sequential_profile_stabilization",
    "profile_scale_calibrated_sequential_profile_stabilization",
    "profile_scale_diversity_calibrated_sequential_profile_stabilization",
    "profile_scale_frontier_diversity_calibrated_sequential_profile_stabilization",
    "profile_scale_coverage_frontier_diversity_calibrated_sequential_profile_stabilization",
    "profile_scale_coverage_prep_frontier_diversity_calibrated_sequential_profile_stabilization",
    "profile_scale_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization",
    "profile_scale_branch_stable_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization",
    "profile_scale_branch_diversity_recovery_frontier_calibrated_sequential_profile_stabilization",
    "profile_scale_collapsed_profile_binding_frontier_calibrated_sequential_profile_stabilization",
    "profile_scale_remaining_profile_binding_frontier_calibrated_sequential_profile_stabilization",
    "profile_scale_owner_paraphrase_binding_frontier_calibrated_sequential_profile_stabilization",
    "profile_scale_memory_consolidation_frontier_calibrated_sequential_profile_stabilization",
    "profile_scale_memory_consolidation_missing_first_token_frontier_calibrated_sequential_profile_stabilization",
    "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_calibrated_sequential_profile_stabilization",
    "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_calibrated_sequential_profile_stabilization",
}


def direct_answer_update_parameters(
    model: Any,
    train_top_layer_only: bool,
    freeze_output_bias: bool,
) -> list[Scalar]:
    model.freeze_lower_layers_for_updates = (
        train_top_layer_only and model.config.num_layers > 1
    )
    params = model.top_layer_parameters() if train_top_layer_only else model.parameters()
    if freeze_output_bias:
        params = exclude_scalars(params, model.bout)
    return params


def restore_direct_answer_update_state(
    model_class: Any,
    model_payload: dict[str, Any],
    optimizer_payload: dict[str, Any],
    current_tokenizer: CharTokenizer,
    train_top_layer_only: bool,
    freeze_output_bias: bool,
) -> tuple[Any, CharTokenizer, ScalarOptimizer, list[Scalar]]:
    restored_model, restored_tokenizer = model_class.from_dict(model_payload)
    tokenizer = restored_tokenizer if restored_tokenizer is not None else current_tokenizer
    optimizer = ScalarOptimizer.from_dict(optimizer_payload)
    restored_model.active_optimizer = optimizer
    params = direct_answer_update_parameters(
        restored_model,
        train_top_layer_only,
        freeze_output_bias,
    )
    return restored_model, tokenizer, optimizer, params


def record_direct_update_guard_rejection_attempt(
    direct_answer_update_guard: dict[str, Any],
    direct_baseline: dict[str, Any],
    direct_step: int,
    probe_snapshot: dict[str, Any],
    learning_rate_scale: float,
    update_shape: str = "direct",
) -> None:
    direct_answer_update_guard["rejected_attempts"] += 1
    scale_key = f"{learning_rate_scale:g}"
    rejected_scale_counts = direct_answer_update_guard[
        "rejected_learning_rate_scale_counts"
    ]
    if isinstance(rejected_scale_counts, dict):
        rejected_scale_counts[scale_key] = int(rejected_scale_counts.get(scale_key, 0)) + 1
    rejected_shape_counts = direct_answer_update_guard["rejected_update_shape_counts"]
    if isinstance(rejected_shape_counts, dict):
        rejected_shape_counts[update_shape] = int(rejected_shape_counts.get(update_shape, 0)) + 1
    floor_diagnostics = branch_diversity_snapshot_target_coverage_diagnostics(
        probe_snapshot,
        direct_baseline,
    )
    violation_profile_counts = direct_answer_update_guard[
        "rejected_violation_profile_counts"
    ]
    if isinstance(violation_profile_counts, dict):
        for violation in floor_diagnostics["violations"]:
            profile = str(violation["profile"])
            violation_profile_counts[profile] = int(violation_profile_counts.get(profile, 0)) + 1
    worst_deficit = float(floor_diagnostics["worst_deficit"])
    if worst_deficit > float(
        direct_answer_update_guard["worst_rejected_coverage_deficit"]
    ):
        direct_answer_update_guard["worst_rejected_coverage_deficit"] = worst_deficit
        direct_answer_update_guard["worst_rejected_coverage_violation"] = (
            floor_diagnostics["worst_violation"]
        )
    diagnostic_sample = direct_answer_update_guard["rejected_floor_diagnostic_sample"]
    if isinstance(diagnostic_sample, list) and len(diagnostic_sample) < 12:
        diagnostic_sample.append(
            {
                "step": direct_step,
                "learning_rate_scale": learning_rate_scale,
                "update_shape": update_shape,
                "preserved": floor_diagnostics["preserved"],
                "violating_profile_count": floor_diagnostics[
                    "violating_profile_count"
                ],
                "worst_deficit": floor_diagnostics["worst_deficit"],
                "worst_violation": floor_diagnostics["worst_violation"],
                "violations": floor_diagnostics["violations"][:5],
            }
        )
    rejected_sample = direct_answer_update_guard["rejected_step_sample"]
    if isinstance(rejected_sample, list) and len(rejected_sample) < 12:
        rejected_sample.append(
            {
                "step": direct_step,
                "learning_rate_scale": learning_rate_scale,
                "update_shape": update_shape,
                "coverage": branch_diversity_snapshot_target_coverage_by_profile(
                    probe_snapshot
                ),
                "floor_diagnostics": {
                    "preserved": floor_diagnostics["preserved"],
                    "violating_profile_count": floor_diagnostics[
                        "violating_profile_count"
                    ],
                    "worst_deficit": floor_diagnostics["worst_deficit"],
                    "worst_violation": floor_diagnostics["worst_violation"],
                },
            }
        )


def record_direct_update_guard_acceptance(
    direct_answer_update_guard: dict[str, Any],
    learning_rate_scale: float,
    update_shape: str = "direct",
) -> None:
    direct_answer_update_guard["accepted_steps"] += 1
    direct_answer_update_guard["accepted_attempts"] += 1
    if update_shape == "repaired":
        direct_answer_update_guard["repaired_steps"] += 1
        direct_answer_update_guard["repaired_attempts"] += 1
    if update_shape in STABILIZATION_UPDATE_SHAPES:
        direct_answer_update_guard["stabilized_steps"] += 1
        direct_answer_update_guard["stabilized_attempts"] += 1
    scale_key = f"{learning_rate_scale:g}"
    scale_counts = direct_answer_update_guard["accepted_learning_rate_scale_counts"]
    if isinstance(scale_counts, dict):
        scale_counts[scale_key] = int(scale_counts.get(scale_key, 0)) + 1
    shape_counts = direct_answer_update_guard["accepted_update_shape_counts"]
    if isinstance(shape_counts, dict):
        shape_counts[update_shape] = int(shape_counts.get(update_shape, 0)) + 1


def apply_direct_update_guard_probe(
    *,
    direct_answer_update_guard: dict[str, Any],
    direct_baseline: dict[str, Any],
    direct_step: int,
    direct_snapshot_recorder: Any,
    pre_update_model_payload: dict[str, Any] | None,
    pre_update_optimizer_payload: dict[str, Any] | None,
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
) -> bool:
    direct_answer_update_guard["checked_steps"] += 1
    direct_answer_update_guard["attempted_updates"] += 1
    probe_snapshot = direct_snapshot_recorder.record(
        direct_step,
        None,
        {
            "baseline_floor_update_guard_probe": True,
            "learning_rate_scale": 1.0,
        },
    )
    if branch_diversity_snapshot_preserves_target_coverage(
        probe_snapshot,
        direct_baseline,
    ):
        record_direct_update_guard_acceptance(direct_answer_update_guard, 1.0)
        return True
    direct_answer_update_guard["rejected_steps"] += 1
    record_direct_update_guard_rejection_attempt(
        direct_answer_update_guard,
        direct_baseline,
        direct_step,
        probe_snapshot,
        1.0,
    )
    if pre_update_model_payload is not None and pre_update_optimizer_payload is not None:
        restore_direct_update_state(
            pre_update_model_payload,
            pre_update_optimizer_payload,
        )
    return False
