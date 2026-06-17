"""Binding and memory summary fields for direct-answer replay plans."""

from __future__ import annotations

import argparse
from typing import Any

from transformer_remaining_profile_binding import source_profile_label
import transformer_direct_modes as modes


def attach_binding_summary(
    replay_plan: dict[str, Any],
    args: argparse.Namespace,
    flags: dict[str, bool],
    memory: Any,
    remaining_targets: list[str],
    remaining_source_labels: list[str],
) -> None:
    if not flags["profile_scale_remaining_profile_binding_frontier_stabilization_active"]:
        return
    replay_plan["remaining_profile_binding_target_profiles"] = remaining_targets
    replay_plan["remaining_profile_binding_source_labels"] = remaining_source_labels
    replay_plan["remaining_profile_binding_source_profiles"] = [
        profile
        for profile in sorted(replay_plan["profiles"])
        if source_profile_label(profile) in set(remaining_source_labels)
    ]
    if flags["profile_scale_owner_paraphrase_binding_frontier_stabilization_active"]:
        replay_plan["owner_paraphrase_binding_target_profiles"] = list(
            modes.BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES
        )
        replay_plan["owner_paraphrase_binding_preserved_profiles"] = list(
            modes.BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES
        )
        replay_plan["owner_paraphrase_binding_source_labels"] = remaining_source_labels
        replay_plan["owner_paraphrase_binding_source_profiles"] = replay_plan[
            "remaining_profile_binding_source_profiles"
        ]
    if flags["profile_scale_memory_consolidation_frontier_stabilization_active"]:
        attach_memory_consolidation_summary(replay_plan, args, flags, memory)


def attach_memory_consolidation_summary(
    replay_plan: dict[str, Any],
    args: argparse.Namespace,
    flags: dict[str, bool],
    memory: Any,
) -> None:
    remaining_collapsed = flags[
        "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active"
    ]
    profile_specific = flags[
        "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active"
    ]
    replay_plan.update(
        {
            "memory_consolidation_source_plan": str(memory.source_plan_path),
            "memory_consolidation_source_plan_summary": memory.source_plan_summary,
            "memory_consolidation_target_profiles": memory.target_profiles,
            "memory_consolidation_top_priority_profiles": memory.top_priority_profiles,
            "memory_consolidation_collapsed_memory_backed_profiles": (
                memory.collapsed_memory_backed_profiles
            ),
            "memory_consolidation_max_profiles": args.memory_consolidation_max_profiles,
            "memory_consolidation_consumed_profile_count": len(memory.target_profiles),
            "memory_consolidation_missing_first_token_target_tokens": (
                memory.missing_first_token_values
            ),
            "memory_consolidation_missing_first_token_target_ids": (
                memory.missing_first_token_ids
            ),
            "memory_consolidation_missing_first_token_learning_rate_scales": (
                list(modes.BASELINE_FLOOR_MISSING_FIRST_TOKEN_LEARNING_RATE_SCALES)
                if flags[
                    "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active"
                ]
                else []
            ),
            "memory_consolidation_remaining_collapsed_target_profiles": (
                list(memory.target_profiles) if remaining_collapsed else []
            ),
            "memory_consolidation_remaining_collapsed_source_profiles": (
                list(memory.collapsed_memory_backed_profiles)
                if remaining_collapsed
                else []
            ),
            "memory_consolidation_profile_specific_missing_first_token_target_map": (
                memory.profile_specific_missing_first_token_target_map
                if profile_specific
                else {}
            ),
        }
    )
