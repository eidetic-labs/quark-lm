"""Assembled branch-binding objectives for TinyTransformerLM."""

from __future__ import annotations

from transformer_lm_branch_pairwise_binding import (
    TransformerBranchPairwiseBindingMixin,
)
from transformer_lm_branch_target_binding import TransformerBranchTargetBindingMixin
from transformer_lm_branch_target_coverage import (
    TransformerBranchTargetCoverageMixin,
)


class TransformerBranchBindingMixin(
    TransformerBranchPairwiseBindingMixin,
    TransformerBranchTargetBindingMixin,
    TransformerBranchTargetCoverageMixin,
):
    """Compatibility mixin combining focused branch-binding objective families."""
