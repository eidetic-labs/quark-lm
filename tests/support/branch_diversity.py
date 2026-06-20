"""Branch-diversity diagnostics and replay fixtures for transformer tests."""

from __future__ import annotations

from branch_diversity_diagnostics import branch_routing_audit_summary
from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
    branch_diversity_snapshot_target_coverage_diagnostics,
)
from branch_diversity_snapshots import (
    branch_diversity_profile_delta_has_coverage_gain,
    branch_diversity_snapshot_collapsed_profile_names,
    branch_diversity_snapshot_profile_diversity_delta,
    branch_diversity_snapshot_score,
    branch_diversity_snapshot_score_improved,
)
from replay_plan import branch_replay_plan
from transformer_branch_diversity_summary import summarize_branch_diversity_target
from transformer_branch_logit_diagnostics import (
    direct_answer_branch_logit_prior_profile,
)
from transformer_branch_profiles import direct_answer_branch_profile
from transformer_branch_representation_profiles import (
    direct_answer_branch_representation_profile,
)

__all__ = [
    "branch_diversity_profile_delta_has_coverage_gain",
    "branch_diversity_snapshot_collapsed_profile_names",
    "branch_diversity_snapshot_preserves_target_coverage",
    "branch_diversity_snapshot_profile_diversity_delta",
    "branch_diversity_snapshot_score",
    "branch_diversity_snapshot_score_improved",
    "branch_diversity_snapshot_target_coverage_delta",
    "branch_diversity_snapshot_target_coverage_diagnostics",
    "branch_replay_plan",
    "branch_routing_audit_summary",
    "direct_answer_branch_logit_prior_profile",
    "direct_answer_branch_profile",
    "direct_answer_branch_representation_profile",
    "summarize_branch_diversity_target",
]
