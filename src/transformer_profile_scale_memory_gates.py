"""Memory-consolidation profile-scale experiment gates."""

from __future__ import annotations

from typing import Any

from transformer_experiment_modes import (
    PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
)
from transformer_profile_scale_gate_common import required_gate


MEMORY_CONSOLIDATION_PROFILE_SCALE_MODES = {
    PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
}

MISSING_FIRST_TOKEN_PROFILE_SCALE_MODES = {
    PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
}

REMAINING_COLLAPSED_PROFILE_SCALE_MODES = {
    PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
}


def profile_scale_memory_gate(direct_answer_mode: str) -> dict[str, Any] | None:
    if direct_answer_mode not in MEMORY_CONSOLIDATION_PROFILE_SCALE_MODES:
        return None
    mode_flags = _memory_consolidation_flags(direct_answer_mode)
    return required_gate(
        "baseline_floor_profile_scale_memory_consolidation_"
        + _memory_consolidation_name_suffix(*mode_flags),
        _memory_consolidation_rule(*mode_flags),
    )


def _memory_consolidation_flags(direct_answer_mode: str) -> tuple[bool, bool, bool]:
    missing_first_token_mode = direct_answer_mode in MISSING_FIRST_TOKEN_PROFILE_SCALE_MODES
    remaining_collapsed_mode = direct_answer_mode in REMAINING_COLLAPSED_PROFILE_SCALE_MODES
    profile_specific_mode = (
        direct_answer_mode
        == PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE
    )
    return (
        missing_first_token_mode,
        remaining_collapsed_mode,
        profile_specific_mode,
    )


def _memory_consolidation_name_suffix(
    missing_first_token_mode: bool,
    remaining_collapsed_mode: bool,
    profile_specific_mode: bool,
) -> str:
    if profile_specific_mode:
        return (
            "remaining_collapsed_profile_specific_missing_first_token_frontier_"
            "calibrated_sequential_stabilization_screen"
        )
    if remaining_collapsed_mode:
        return (
            "remaining_collapsed_missing_first_token_frontier_calibrated_"
            "sequential_stabilization_screen"
        )
    if missing_first_token_mode:
        return (
            "missing_first_token_frontier_calibrated_sequential_"
            "stabilization_screen"
        )
    return "frontier_calibrated_sequential_stabilization_screen"


def _memory_consolidation_rule(
    missing_first_token_mode: bool,
    remaining_collapsed_mode: bool,
    profile_specific_mode: bool,
) -> str:
    rule_parts = [
        "Run records memory-consolidation frontier activation, ",
        "the declared source memory_consolidation_plan path, ",
        "source-plan summary, consumed target profiles, top priority profiles, ",
        "collapsed memory-backed profiles, ",
    ]
    if missing_first_token_mode:
        rule_parts.extend(
            [
                "missing first-token target maps, missing first-token candidates, ",
                "attempts, acceptances, fallback acceptances, rejections, ",
                "rejection reasons, profile-diversity deltas, ",
            ]
        )
    if remaining_collapsed_mode:
        rule_parts.append(
            "remaining collapsed target profiles, remaining collapsed source profiles, "
        )
    if profile_specific_mode:
        rule_parts.append("profile-specific missing first-token target map, ")
    rule_parts.append(
        "prioritized attempts, prioritized acceptances and rejections, "
        "preservation checks, collapsed-profile binding candidates, attempts, "
        "acceptances, fallback acceptances, rejection reasons, "
        "profile-diversity deltas, update-shape counts, replay plan, "
        "branch-context gate, coverage floor, diversity target, recipe, verifier, "
        "and constraint-first promotion artifacts."
    )
    return "".join(rule_parts)
