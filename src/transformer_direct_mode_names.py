"""Direct-answer mode names and shared type aliases."""

from __future__ import annotations


ANSWER_TERMINATOR = "\n"
ReplayPredictionOverrides = dict[tuple[tuple[int, ...], int, str], int]
BASELINE_ANCHORED_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-anchored-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_GATED_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_ADAPTIVE_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_REPAIRED_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_OBJECTIVE_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood"
)
BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-"
    "profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-"
    "sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-prep-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-recovery-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-frontier-profile-scale-calibrated-sequential-"
    "profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "frontier-profile-scale-calibrated-sequential-profile-stabilization-"
    "unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-frontier-profile-scale-calibrated-"
    "sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-missing-first-token-frontier-profile-"
    "scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-remaining-collapsed-missing-first-token-"
    "frontier-profile-scale-calibrated-sequential-profile-stabilization-"
    "unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-remaining-collapsed-profile-specific-"
    "missing-first-token-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)

__all__ = tuple(
    name
    for name in globals()
    if name.isupper() or name == "ReplayPredictionOverrides"
)
