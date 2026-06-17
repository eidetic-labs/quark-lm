from __future__ import annotations

from support.branch_replay_coverage_fixtures import BranchBatch
from support.core import TinyTransformerLM


def train_deficit_focus_comparison_steps(
    baseline_model: TinyTransformerLM,
    deficit_model: TinyTransformerLM,
    branch_batch: BranchBatch,
    replay_branches: BranchBatch,
    *,
    repeat: int = 48,
) -> None:
    for _ in range(repeat):
        baseline_model.train_step_with_branch_context_replay_coverage(
            branch_batch,
            replay_branches,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            hard_negative_count=5,
        )
        deficit_model.train_step_with_branch_context_replay_coverage(
            branch_batch,
            replay_branches,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            hard_negative_count=5,
            focus_uncovered_targets=True,
        )


def train_preserving_deficit_steps(
    deficit_only_model: TinyTransformerLM,
    preserving_model: TinyTransformerLM,
    branch_batch: BranchBatch,
    replay_branches: BranchBatch,
    *,
    repeat: int = 48,
) -> None:
    for _ in range(repeat):
        deficit_only_model.train_step_with_branch_context_replay_coverage(
            branch_batch,
            replay_branches,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            hard_negative_count=5,
            focus_uncovered_targets=True,
        )
        preserving_model.train_step_with_branch_context_replay_coverage(
            branch_batch,
            replay_branches,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            hard_negative_count=5,
            focus_uncovered_targets=True,
            preserve_predicted_target_coverage=True,
            balance_deficit_targets=True,
        )
