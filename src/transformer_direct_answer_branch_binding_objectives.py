"""Compatibility facade for branch binding and target-coverage objectives."""

from __future__ import annotations

from transformer_direct_answer_branch_pairwise_binding_objectives import (
    train_direct_answer_branch_bidirectional_binding_unlikelihood,
    train_direct_answer_branch_coverage_binding_unlikelihood,
)
from transformer_direct_answer_branch_representation_objectives import (
    train_direct_answer_branch_output_binding_unlikelihood,
    train_direct_answer_branch_representation_contrast_unlikelihood,
)
from transformer_direct_answer_branch_target_coverage_objectives import (
    train_direct_answer_branch_target_diversity_unlikelihood,
    train_direct_answer_branch_target_set_coverage_unlikelihood,
)
from transformer_direct_answer_branch_target_replay_objectives import (
    train_direct_answer_branch_target_replay_coverage_unlikelihood,
)


__all__ = [
    "train_direct_answer_branch_bidirectional_binding_unlikelihood",
    "train_direct_answer_branch_coverage_binding_unlikelihood",
    "train_direct_answer_branch_output_binding_unlikelihood",
    "train_direct_answer_branch_representation_contrast_unlikelihood",
    "train_direct_answer_branch_target_diversity_unlikelihood",
    "train_direct_answer_branch_target_replay_coverage_unlikelihood",
    "train_direct_answer_branch_target_set_coverage_unlikelihood",
]
