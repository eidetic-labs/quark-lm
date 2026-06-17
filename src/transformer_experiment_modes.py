"""Profile-aware direct-answer modes for transformer experiments."""

from __future__ import annotations

from typing import Any


PROFILE_SCALE_DIVERSITY_MODE = (
    "branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-"
    "sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COVERAGE_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-prep-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-recovery-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-frontier-profile-scale-calibrated-sequential-"
    "profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "frontier-profile-scale-calibrated-sequential-profile-stabilization-"
    "unlikelihood"
)
PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-frontier-profile-scale-calibrated-"
    "sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-missing-first-token-frontier-profile-"
    "scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-remaining-collapsed-missing-first-token-"
    "frontier-profile-scale-calibrated-sequential-profile-stabilization-"
    "unlikelihood"
)
PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-memory-consolidation-remaining-collapsed-profile-specific-"
    "missing-first-token-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
PROFILE_AWARE_DIRECT_ANSWER_MODES = {
    "branch-context-profile-coverage-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-context-profile-baseline-floor-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    PROFILE_SCALE_DIVERSITY_MODE,
    PROFILE_SCALE_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
}


def is_profile_aware_direct_answer_mode(mode: str) -> bool:
    return mode in PROFILE_AWARE_DIRECT_ANSWER_MODES


def direct_answer_is_profile_aware(args: Any) -> bool:
    return (
        getattr(args, "direct_answer_steps", 0) > 0
        and is_profile_aware_direct_answer_mode(
            getattr(args, "direct_answer_mode", "")
        )
    )
