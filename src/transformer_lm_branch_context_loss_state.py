"""Loss accumulators for branch-context replay objectives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from autograd import Scalar


def scalar_zero() -> Scalar:
    return Scalar(0.0)


@dataclass
class BranchContextLossState:
    branch_loss: Scalar = field(default_factory=scalar_zero)
    replay_coverage_loss: Scalar = field(default_factory=scalar_zero)
    replay_ownership_loss: Scalar = field(default_factory=scalar_zero)
    deficit_target_loss: Scalar = field(default_factory=scalar_zero)
    deficit_target_count: int = 0
    deficit_target_losses_by_target: dict[tuple[str, int], Scalar] = field(
        default_factory=dict
    )
    deficit_target_counts_by_target: Counter[tuple[str, int]] = field(
        default_factory=Counter
    )
    profile_target_share_losses_by_target: dict[tuple[str, int], Scalar] = field(
        default_factory=dict
    )
    profile_target_share_counts_by_target: Counter[tuple[str, int]] = field(
        default_factory=Counter
    )
    prompt_target_margin_loss: Scalar = field(default_factory=scalar_zero)
    prompt_target_margin_count: int = 0
    coverage_preservation_losses_by_target: dict[tuple[str, int], Scalar] = field(
        default_factory=dict
    )
    coverage_preservation_counts_by_target: Counter[tuple[str, int]] = field(
        default_factory=Counter
    )
    covered_anchor_loss: Scalar = field(default_factory=scalar_zero)
    covered_anchor_count: int = 0
    covered_anchor_losses_by_target: dict[tuple[str, int], Scalar] = field(
        default_factory=dict
    )
    covered_anchor_counts_by_target: Counter[tuple[str, int]] = field(
        default_factory=Counter
    )
    floor_preservation_loss: Scalar = field(default_factory=scalar_zero)
    floor_preservation_count: int = 0
    floor_preservation_losses_by_target: dict[tuple[str, int], Scalar] = field(
        default_factory=dict
    )
    floor_preservation_counts_by_target: Counter[tuple[str, int]] = field(
        default_factory=Counter
    )
