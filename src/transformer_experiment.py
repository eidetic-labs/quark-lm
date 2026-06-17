"""Compatibility facade for transformer experiment surfaces."""

from __future__ import annotations

from transformer_experiment_constants import (
    TRAINING_DATA_DESCRIPTION,
    TRANSFORMER_RECIPE_VERSION,
)
from transformer_experiment_decision import transformer_experiment_decision
from transformer_experiment_gates import (
    parse_experiment_gate,
    transformer_experiment_acceptance_gates,
)
from transformer_experiment_intent import transformer_experiment_intent
from transformer_experiment_modes import (
    PROFILE_AWARE_DIRECT_ANSWER_MODES,
    PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_DIVERSITY_MODE,
    PROFILE_SCALE_FRONTIER_MODE,
    PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE,
    direct_answer_is_profile_aware,
    is_profile_aware_direct_answer_mode,
)
from transformer_experiment_recipe import (
    transformer_training_recipe,
    transformer_training_recipe_id,
)
from transformer_run_artifacts import TransformerRunArtifacts


__all__ = [
    "PROFILE_AWARE_DIRECT_ANSWER_MODES",
    "PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE",
    "PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE",
    "PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE",
    "PROFILE_SCALE_COVERAGE_FRONTIER_MODE",
    "PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE",
    "PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE",
    "PROFILE_SCALE_DIVERSITY_MODE",
    "PROFILE_SCALE_FRONTIER_MODE",
    "PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_MODE",
    "PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE",
    "PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE",
    "PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE",
    "PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE",
    "PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE",
    "TRAINING_DATA_DESCRIPTION",
    "TRANSFORMER_RECIPE_VERSION",
    "TransformerRunArtifacts",
    "direct_answer_is_profile_aware",
    "is_profile_aware_direct_answer_mode",
    "parse_experiment_gate",
    "transformer_experiment_acceptance_gates",
    "transformer_experiment_decision",
    "transformer_experiment_intent",
    "transformer_training_recipe",
    "transformer_training_recipe_id",
]
