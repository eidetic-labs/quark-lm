"""Memory-consolidation setup for direct-answer training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizer import CharTokenizer
import transformer_direct_modes as modes
from transformer_memory_plan_helpers import (
    memory_consolidation_missing_first_token_values,
    memory_consolidation_source_plan_targets,
    missing_first_token_ids_by_profile,
    profile_specific_missing_first_token_target_map,
)


@dataclass
class DirectMemoryConsolidationSetup:
    source_plan_path: Path | None
    source_plan_summary: dict[str, Any]
    target_profiles: list[str]
    top_priority_profiles: list[str]
    collapsed_memory_backed_profiles: list[str]
    missing_first_token_values: dict[str, list[str]]
    missing_first_token_ids: dict[str, list[int]]
    profile_specific_missing_first_token_target_map: dict[str, list[str]]


def prepare_direct_memory_consolidation(
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
    flags: dict[str, bool],
) -> DirectMemoryConsolidationSetup:
    if not flags["profile_scale_memory_consolidation_frontier_stabilization_active"]:
        return DirectMemoryConsolidationSetup(None, {}, [], [], [], {}, {}, {})
    if args.memory_consolidation_source_plan is None:
        raise ValueError(
            "memory consolidation mode requires --memory-consolidation-source-plan"
        )
    source_plan_path = args.memory_consolidation_source_plan
    with source_plan_path.open("r", encoding="utf-8") as handle:
        source_plan = json.load(handle)
    summary, targets, top_priority, collapsed_backed = (
        memory_consolidation_source_plan_targets(
            source_plan,
            args.memory_consolidation_max_profiles,
            require_collapsed_targets=flags[
                "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active"
            ],
        )
    )
    missing_values = memory_consolidation_missing_first_token_values(
        source_plan,
        targets,
    )
    missing_ids = missing_first_token_ids_by_profile(tokenizer, missing_values)
    return DirectMemoryConsolidationSetup(
        source_plan_path,
        summary,
        targets,
        top_priority,
        collapsed_backed,
        missing_values,
        missing_ids,
        profile_specific_missing_first_token_target_map(targets, missing_ids),
    )


def remaining_profile_binding_targets(
    memory: DirectMemoryConsolidationSetup,
    flags: dict[str, bool],
) -> list[str]:
    if flags["profile_scale_memory_consolidation_frontier_stabilization_active"]:
        return list(memory.target_profiles)
    if flags["profile_scale_owner_paraphrase_binding_frontier_stabilization_active"]:
        return list(modes.BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES)
    return list(modes.BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_PROFILES)


def direct_memory_field_kwargs(
    memory: DirectMemoryConsolidationSetup,
) -> dict[str, Any]:
    return {
        "direct_memory_consolidation_source_plan_path": memory.source_plan_path,
        "direct_memory_consolidation_source_plan_summary": memory.source_plan_summary,
        "direct_memory_consolidation_target_profiles": memory.target_profiles,
        "direct_memory_consolidation_top_priority_profiles": memory.top_priority_profiles,
        "direct_memory_consolidation_collapsed_memory_backed_profiles": (
            memory.collapsed_memory_backed_profiles
        ),
        "direct_memory_consolidation_missing_first_token_values": (
            memory.missing_first_token_values
        ),
        "direct_memory_consolidation_missing_first_token_ids": (
            memory.missing_first_token_ids
        ),
        "direct_memory_consolidation_profile_specific_missing_first_token_target_map": (
            memory.profile_specific_missing_first_token_target_map
        ),
    }
