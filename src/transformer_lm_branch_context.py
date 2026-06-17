"""Profile-aware branch-context replay objectives for TinyTransformerLM."""

from __future__ import annotations

from autograd import Scalar
from replay_plan import BranchReplayRecord
from transformer_lm_branch_context_training import train_branch_context_replay_coverage


class TransformerBranchContextMixin:
    def train_step_with_branch_context_replay_coverage(
        self,
        branches: list[BranchReplayRecord],
        replay_branches: list[BranchReplayRecord],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        replay_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
        preserve_covered_targets: bool = False,
        balance_covered_target_anchors: bool = False,
        focus_uncovered_targets: bool = False,
        preserve_predicted_target_coverage: bool = False,
        balance_deficit_targets: bool = False,
        profile_aware_targets: bool = False,
        balance_profile_target_shares: bool = False,
        enforce_prompt_target_margins: bool = False,
        floor_preservation_branches: list[BranchReplayRecord] | None = None,
        floor_preservation_weight: float = 0.0,
        balance_floor_preservation_targets: bool = False,
    ) -> float:
        return train_branch_context_replay_coverage(
            self,
            branches,
            replay_branches,
            learning_rate,
            negative_weight,
            positive_weight,
            replay_weight,
            hard_negative_count,
            params,
            preserve_covered_targets,
            balance_covered_target_anchors,
            focus_uncovered_targets,
            preserve_predicted_target_coverage,
            balance_deficit_targets,
            profile_aware_targets,
            balance_profile_target_shares,
            enforce_prompt_target_margins,
            floor_preservation_branches,
            floor_preservation_weight,
            balance_floor_preservation_targets,
        )
