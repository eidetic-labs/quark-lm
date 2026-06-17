from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_char_metrics import continuation_nll
from neural_char_ops import context_before
from tokenizer import CharTokenizer
from answer_model import AnswerExample
from branch_diversity_diagnostics import branch_routing_audit_summary
from branch_diversity_snapshots import (
    branch_diversity_snapshot_collapsed_profile_names,
    branch_diversity_profile_delta_has_coverage_gain,
    branch_diversity_snapshot_profile_diversity_delta,
    branch_diversity_snapshot_score,
)
from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
    branch_diversity_snapshot_target_coverage_diagnostics,
)
from replay_plan import branch_replay_plan
from transformer_answer_evaluation import evaluate_answer_records
from transformer_answer_generator import (
    TransformerGuidedAnswerGenerator,
    build_transformer_answer_generator,
    evaluate_answer_generator_records,
    transformer_answer_generator_training_pool,
)
from transformer_answer_selector import AnswerCandidateSelector, build_answer_selector
from transformer_answer_training_helpers import transformer_direct_answer_training_pool
from transformer_answer_training_steps import (
    sampled_choice_candidates,
    train_answer_char,
    train_answer_mixed_step,
)
from transformer_baseline_floor_anchor_selection import (
    baseline_floor_anchor_profile_groups,
    baseline_floor_objective_anchor_batch,
)
from transformer_baseline_floor_anchor_batches import (
    baseline_floor_repair_anchor_records,
)
from transformer_baseline_floor_anchor_profiles import (
    baseline_floor_anchor_profile_target_count,
)
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_batch,
)
from transformer_direct_answer_repair_selection import direct_answer_hard_branch_contrast
from transformer_branch_diversity_summary import summarize_branch_diversity_target
from transformer_branch_logit_diagnostics import (
    direct_answer_branch_logit_prior_profile,
)
from transformer_branch_profiles import (
    direct_answer_branch_profile,
)
from transformer_branch_representation_profiles import (
    direct_answer_branch_representation_profile,
)
from transformer_char_model import train_transformer_answers
from transformer_cli import parse_args
from transformer_direct_answer_batches import (
    direct_answer_branch_batch,
    direct_answer_branch_diversity_batch,
    direct_answer_dominant_branch_prediction,
    direct_answer_profiled_branch_batch,
    direct_answer_profiled_replay_records,
    direct_answer_target_balanced_branch_batch,
    direct_answer_target_balanced_branch_diversity_batch,
)
from transformer_direct_answer_branch_basic_objectives import (
    train_direct_answer_branch_batch_contrast_unlikelihood,
    train_direct_answer_branch_collapse_unlikelihood,
    train_direct_answer_branch_diversity_unlikelihood,
    train_direct_answer_branch_hidden_projection_margin_unlikelihood,
    train_direct_answer_branch_target_margin_unlikelihood,
    train_direct_answer_branch_target_softmax_unlikelihood,
)
from transformer_direct_answer_branch_binding_objectives import (
    train_direct_answer_branch_bidirectional_binding_unlikelihood,
    train_direct_answer_branch_coverage_binding_unlikelihood,
    train_direct_answer_branch_output_binding_unlikelihood,
    train_direct_answer_branch_representation_contrast_unlikelihood,
    train_direct_answer_branch_target_diversity_unlikelihood,
    train_direct_answer_branch_target_replay_coverage_unlikelihood,
    train_direct_answer_branch_target_set_coverage_unlikelihood,
)
from transformer_direct_answer_branch_contrast_objectives import (
    train_direct_answer_branch_contrast_unlikelihood,
    train_direct_answer_branch_rank_margin_unlikelihood,
    train_direct_answer_branch_span_contrast_unlikelihood,
    train_direct_answer_branch_topk_softmax_unlikelihood,
    train_direct_answer_hard_branch_contrast_unlikelihood,
)
from transformer_direct_answer_context_replay_objective import (
    train_direct_answer_branch_context_replay_coverage_unlikelihood,
)
from transformer_direct_answer_core import (
    answer_sequence_nll,
    direct_answer_branch_context,
    direct_answer_branch_span_position,
    direct_answer_branch_target_ids,
    direct_answer_lesson,
    direct_answer_sequence_nll,
)
from transformer_direct_answer_evaluation import (
    audit_direct_answer_branch_context_coverage,
    audit_prompt_context_coverage,
    evaluate_direct_answer_records,
    summarize_branch_context_coverage_gate,
)
from transformer_direct_answer_repair_objectives import (
    train_direct_answer_balanced_repair_unlikelihood,
    train_direct_answer_branch_repair_unlikelihood,
    train_direct_answer_branch_span_repair_unlikelihood,
    train_direct_answer_early_stop_unlikelihood,
    train_direct_answer_generated_prefix_recovery_unlikelihood,
    train_direct_answer_loop_escape_unlikelihood,
    train_direct_answer_repeat_loop_unlikelihood,
    train_direct_answer_rollout_unlikelihood,
    train_direct_answer_sequence_repair_unlikelihood,
)
from transformer_direct_answer_repairs import (
    direct_answer_branch_repair_error,
    direct_answer_branch_span_repair_error,
    direct_answer_early_stop_error,
    direct_answer_first_error,
    direct_answer_generated_prefix_recovery,
    direct_answer_repeat_loop_error,
    direct_answer_rollout_error,
    direct_answer_sequence_repair_errors,
    has_repeated_suffix,
    train_direct_answer_first_error,
    train_direct_answer_first_error_unlikelihood,
    train_direct_answer_lesson,
)
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_eval import score_transformer_records
from transformer_experiment import (
    transformer_experiment_decision,
    transformer_experiment_intent,
    transformer_training_recipe_id,
)
from transformer_math import exclude_scalars, flatten_scalars, generation_distribution
from transformer_memory_plan_helpers import (
    memory_consolidation_missing_first_token_values,
    memory_consolidation_source_plan_targets,
    missing_first_token_anchor_batch,
    missing_first_token_ids_by_profile,
    profile_specific_missing_first_token_target_map,
    profile_specific_missing_first_token_targets,
    remaining_profile_binding_profile_order,
    remaining_profile_binding_source_labels,
)
from transformer_model import GenerationConfig, OptimizationConfig, TransformerConfig
from transformer_optimizer import ScalarOptimizer
from transformer_optimizer import (
    save_optimizer_state,
    load_optimizer_state,
)
from transformer_text_commands import transformer_training_recipe
from transformer_tiny_lm import TinyTransformerLM

__all__ = [
    "ANSWER_TERMINATOR",
    "AnswerCandidateSelector",
    "AnswerExample",
    "CharTokenizer",
    "GenerationConfig",
    "OptimizationConfig",
    "ScalarOptimizer",
    "TinyTransformerLM",
    "TransformerConfig",
    "TransformerGuidedAnswerGenerator",
    "answer_sequence_nll",
    "audit_direct_answer_branch_context_coverage",
    "audit_prompt_context_coverage",
    "baseline_floor_anchor_profile_groups",
    "baseline_floor_anchor_profile_target_count",
    "baseline_floor_objective_anchor_batch",
    "baseline_floor_repair_anchor_records",
    "branch_diversity_profile_delta_has_coverage_gain",
    "branch_diversity_snapshot_collapsed_profile_names",
    "branch_diversity_snapshot_preserves_target_coverage",
    "branch_diversity_snapshot_profile_diversity_delta",
    "branch_diversity_snapshot_score",
    "branch_diversity_snapshot_target_coverage_delta",
    "branch_diversity_snapshot_target_coverage_diagnostics",
    "branch_replay_plan",
    "branch_routing_audit_summary",
    "build_answer_selector",
    "build_transformer_answer_generator",
    "context_before",
    "continuation_nll",
    "direct_answer_branch_batch",
    "direct_answer_branch_context",
    "direct_answer_branch_diversity_batch",
    "direct_answer_branch_logit_prior_profile",
    "direct_answer_branch_profile",
    "direct_answer_branch_repair_error",
    "direct_answer_branch_representation_profile",
    "direct_answer_branch_span_position",
    "direct_answer_branch_span_repair_error",
    "direct_answer_branch_target_ids",
    "direct_answer_dominant_branch_prediction",
    "direct_answer_early_stop_error",
    "direct_answer_first_error",
    "direct_answer_generated_prefix_recovery",
    "direct_answer_hard_branch_contrast",
    "direct_answer_lesson",
    "direct_answer_profiled_branch_batch",
    "direct_answer_profiled_replay_records",
    "direct_answer_repeat_loop_error",
    "direct_answer_rollout_error",
    "direct_answer_sequence_nll",
    "direct_answer_sequence_repair_errors",
    "direct_answer_target_balanced_branch_batch",
    "direct_answer_target_balanced_branch_diversity_batch",
    "evaluate_answer_generator_records",
    "evaluate_answer_records",
    "evaluate_direct_answer_records",
    "exclude_scalars",
    "flatten_scalars",
    "generation_distribution",
    "has_repeated_suffix",
    "load_optimizer_state",
    "memory_consolidation_missing_first_token_values",
    "memory_consolidation_source_plan_targets",
    "missing_first_token_anchor_batch",
    "missing_first_token_ids_by_profile",
    "parse_args",
    "profile_specific_missing_first_token_target_map",
    "profile_specific_missing_first_token_targets",
    "remaining_profile_binding_profile_order",
    "remaining_profile_binding_source_labels",
    "sampled_choice_candidates",
    "save_optimizer_state",
    "score_transformer_records",
    "summarize_branch_context_coverage_gate",
    "summarize_branch_diversity_target",
    "train_answer_char",
    "train_answer_mixed_step",
    "train_direct_answer_balanced_repair_unlikelihood",
    "train_direct_answer_baseline_floor_anchor_batch",
    "train_direct_answer_branch_batch_contrast_unlikelihood",
    "train_direct_answer_branch_bidirectional_binding_unlikelihood",
    "train_direct_answer_branch_collapse_unlikelihood",
    "train_direct_answer_branch_context_replay_coverage_unlikelihood",
    "train_direct_answer_branch_contrast_unlikelihood",
    "train_direct_answer_branch_coverage_binding_unlikelihood",
    "train_direct_answer_branch_diversity_unlikelihood",
    "train_direct_answer_branch_hidden_projection_margin_unlikelihood",
    "train_direct_answer_branch_output_binding_unlikelihood",
    "train_direct_answer_branch_rank_margin_unlikelihood",
    "train_direct_answer_branch_repair_unlikelihood",
    "train_direct_answer_branch_representation_contrast_unlikelihood",
    "train_direct_answer_branch_span_contrast_unlikelihood",
    "train_direct_answer_branch_span_repair_unlikelihood",
    "train_direct_answer_branch_target_diversity_unlikelihood",
    "train_direct_answer_branch_target_margin_unlikelihood",
    "train_direct_answer_branch_target_replay_coverage_unlikelihood",
    "train_direct_answer_branch_target_set_coverage_unlikelihood",
    "train_direct_answer_branch_target_softmax_unlikelihood",
    "train_direct_answer_branch_topk_softmax_unlikelihood",
    "train_direct_answer_early_stop_unlikelihood",
    "train_direct_answer_first_error",
    "train_direct_answer_first_error_unlikelihood",
    "train_direct_answer_generated_prefix_recovery_unlikelihood",
    "train_direct_answer_hard_branch_contrast_unlikelihood",
    "train_direct_answer_lesson",
    "train_direct_answer_loop_escape_unlikelihood",
    "train_direct_answer_repeat_loop_unlikelihood",
    "train_direct_answer_rollout_unlikelihood",
    "train_direct_answer_sequence_repair_unlikelihood",
    "train_transformer_answers",
    "transformer_answer_generator_training_pool",
    "transformer_direct_answer_training_pool",
    "transformer_experiment_decision",
    "transformer_experiment_intent",
    "transformer_training_recipe",
    "transformer_training_recipe_id",
]
