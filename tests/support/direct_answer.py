"""Direct-answer training and repair fixtures used by transformer tests."""

from __future__ import annotations

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
from transformer_direct_answer_batches import (
    direct_answer_branch_batch,
    direct_answer_branch_diversity_batch,
    direct_answer_dominant_branch_prediction,
    direct_answer_target_balanced_branch_batch,
    direct_answer_target_balanced_branch_diversity_batch,
)
from transformer_direct_answer_branch_basic_objectives import (
    train_direct_answer_branch_batch_contrast_unlikelihood,
    train_direct_answer_branch_collapse_unlikelihood,
    train_direct_answer_branch_diversity_unlikelihood,
    train_direct_answer_branch_hidden_projection_margin_unlikelihood,
    train_direct_answer_profile_balanced_branch_hidden_projection_margin_unlikelihood,
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
from transformer_direct_answer_branch_context_evaluation import (
    audit_direct_answer_branch_context_coverage,
    summarize_branch_context_coverage_gate,
)
from transformer_direct_answer_branch_contrast_objectives import (
    train_direct_answer_branch_contrast_unlikelihood,
    train_direct_answer_branch_rank_margin_unlikelihood,
    train_direct_answer_branch_span_contrast_unlikelihood,
    train_direct_answer_branch_topk_softmax_unlikelihood,
    train_direct_answer_hard_branch_contrast_unlikelihood,
)
from transformer_direct_answer_branch_repairs import (
    direct_answer_branch_repair_error,
    direct_answer_branch_span_repair_error,
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
    audit_prompt_context_coverage,
    evaluate_direct_answer_records,
)
from transformer_direct_answer_profiled_batches import (
    direct_answer_profiled_branch_batch,
    direct_answer_profiled_replay_records,
)
from transformer_direct_answer_profile_balanced_batches import (
    direct_answer_profile_balanced_branch_batch,
)
from transformer_direct_answer_repair_discovery import (
    direct_answer_early_stop_error,
    direct_answer_generated_prefix_recovery,
    direct_answer_repeat_loop_error,
    direct_answer_rollout_error,
    direct_answer_sequence_repair_errors,
    has_repeated_suffix,
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
from transformer_direct_answer_repair_selection import direct_answer_hard_branch_contrast
from transformer_direct_answer_repairs import (
    direct_answer_first_error,
    train_direct_answer_first_error,
    train_direct_answer_first_error_unlikelihood,
    train_direct_answer_lesson,
)

__all__ = [
    "AnswerCandidateSelector",
    "TransformerGuidedAnswerGenerator",
    "answer_sequence_nll",
    "audit_direct_answer_branch_context_coverage",
    "audit_prompt_context_coverage",
    "build_answer_selector",
    "build_transformer_answer_generator",
    "direct_answer_branch_batch",
    "direct_answer_branch_context",
    "direct_answer_branch_diversity_batch",
    "direct_answer_branch_repair_error",
    "direct_answer_branch_span_position",
    "direct_answer_branch_span_repair_error",
    "direct_answer_branch_target_ids",
    "direct_answer_dominant_branch_prediction",
    "direct_answer_early_stop_error",
    "direct_answer_first_error",
    "direct_answer_generated_prefix_recovery",
    "direct_answer_hard_branch_contrast",
    "direct_answer_lesson",
    "direct_answer_profile_balanced_branch_batch",
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
    "has_repeated_suffix",
    "sampled_choice_candidates",
    "summarize_branch_context_coverage_gate",
    "train_answer_char",
    "train_answer_mixed_step",
    "train_direct_answer_balanced_repair_unlikelihood",
    "train_direct_answer_branch_batch_contrast_unlikelihood",
    "train_direct_answer_branch_bidirectional_binding_unlikelihood",
    "train_direct_answer_branch_collapse_unlikelihood",
    "train_direct_answer_branch_context_replay_coverage_unlikelihood",
    "train_direct_answer_branch_contrast_unlikelihood",
    "train_direct_answer_branch_coverage_binding_unlikelihood",
    "train_direct_answer_branch_diversity_unlikelihood",
    "train_direct_answer_branch_hidden_projection_margin_unlikelihood",
    "train_direct_answer_profile_balanced_branch_hidden_projection_margin_unlikelihood",
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
    "transformer_answer_generator_training_pool",
    "transformer_direct_answer_training_pool",
]
