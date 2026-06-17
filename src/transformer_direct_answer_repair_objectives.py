"""Compatibility facade for direct-answer repair objectives."""

from __future__ import annotations

from transformer_direct_answer_repair_branch_objectives import (
    train_direct_answer_branch_repair_unlikelihood,
    train_direct_answer_branch_span_repair_unlikelihood,
)
from transformer_direct_answer_repair_negative_objectives import (
    train_direct_answer_early_stop_unlikelihood,
    train_direct_answer_repeat_loop_unlikelihood,
    train_direct_answer_rollout_unlikelihood,
)
from transformer_direct_answer_repair_positive_objectives import (
    train_direct_answer_balanced_repair_unlikelihood,
    train_direct_answer_loop_escape_unlikelihood,
    train_direct_answer_sequence_repair_unlikelihood,
)
from transformer_direct_answer_repair_recovery_objectives import (
    train_direct_answer_generated_prefix_recovery_unlikelihood,
)


__all__ = [
    "train_direct_answer_balanced_repair_unlikelihood",
    "train_direct_answer_branch_repair_unlikelihood",
    "train_direct_answer_branch_span_repair_unlikelihood",
    "train_direct_answer_early_stop_unlikelihood",
    "train_direct_answer_generated_prefix_recovery_unlikelihood",
    "train_direct_answer_loop_escape_unlikelihood",
    "train_direct_answer_repeat_loop_unlikelihood",
    "train_direct_answer_rollout_unlikelihood",
    "train_direct_answer_sequence_repair_unlikelihood",
]
