"""Missing-first-token target planning for memory consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenizer import CharTokenizer
from transformer_remaining_profile_binding import (
    remaining_profile_binding_source_labels,
    source_profile_label,
)


@dataclass(frozen=True)
class MissingFirstTokenTargetPlan:
    target_profiles: list[str]
    target_ids: list[int]
    target_id_set: set[int]


def memory_consolidation_missing_first_token_values(
    source_plan: dict[str, Any],
    target_profiles: list[str] | tuple[str, ...] | set[str],
) -> dict[str, list[str]]:
    target_set = set(target_profiles)
    values_by_profile: dict[str, list[str]] = {}
    priority_records = source_plan.get("profile_priorities", [])
    if not isinstance(priority_records, list):
        return values_by_profile
    for record in priority_records:
        if not isinstance(record, dict):
            continue
        profile = record.get("profile")
        if not isinstance(profile, str) or profile not in target_set:
            continue
        values = _missing_first_token_values(record)
        if values:
            values_by_profile[profile] = values
    return {
        profile: values_by_profile[profile]
        for profile in target_profiles
        if profile in values_by_profile
    }


def missing_first_token_ids_by_profile(
    tokenizer: CharTokenizer,
    values_by_profile: dict[str, list[str]],
) -> dict[str, list[int]]:
    ids_by_profile: dict[str, list[int]] = {}
    for profile, values in values_by_profile.items():
        ids = _single_token_ids(tokenizer, values)
        if ids:
            ids_by_profile[profile] = ids
    return ids_by_profile


def profile_specific_missing_first_token_targets(
    source_profile: str,
    target_profiles: list[str] | tuple[str, ...],
    ids_by_profile: dict[str, list[int]],
) -> list[str]:
    source_label = source_profile_label(source_profile)
    targets: list[str] = []
    for target_profile in target_profiles:
        if not ids_by_profile.get(target_profile):
            continue
        if source_label in set(
            remaining_profile_binding_source_labels([target_profile])
        ):
            targets.append(target_profile)
    return targets


def plan_missing_first_token_targets(
    source_profile: str,
    target_profiles: list[str] | tuple[str, ...],
    ids_by_profile: dict[str, list[int]],
    *,
    profile_specific: bool,
) -> MissingFirstTokenTargetPlan:
    if profile_specific:
        planned_profiles = profile_specific_missing_first_token_targets(
            source_profile,
            target_profiles,
            ids_by_profile,
        )
    else:
        planned_profiles = [
            target_profile
            for target_profile in target_profiles
            if ids_by_profile.get(target_profile)
        ]
    target_id_set = {
        token_id
        for target_profile in planned_profiles
        for token_id in ids_by_profile.get(target_profile, [])
    }
    return MissingFirstTokenTargetPlan(
        target_profiles=planned_profiles,
        target_ids=sorted(target_id_set),
        target_id_set=target_id_set,
    )


def profile_specific_missing_first_token_target_map(
    target_profiles: list[str] | tuple[str, ...],
    ids_by_profile: dict[str, list[int]],
) -> dict[str, list[str]]:
    targets_by_source_label: dict[str, list[str]] = {}
    for target_profile in target_profiles:
        if not ids_by_profile.get(target_profile):
            continue
        for source_label in remaining_profile_binding_source_labels([target_profile]):
            targets_by_source_label.setdefault(source_label, []).append(
                target_profile
            )
    return {
        source_label: targets
        for source_label, targets in sorted(targets_by_source_label.items())
    }


def _missing_first_token_values(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    raw_missing_tokens = record.get("missing_target_tokens", [])
    if not isinstance(raw_missing_tokens, list):
        return values
    for raw_token in raw_missing_tokens:
        value = raw_token.get("value") if isinstance(raw_token, dict) else raw_token
        if not isinstance(value, str) or len(value) != 1 or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _single_token_ids(tokenizer: CharTokenizer, values: list[str]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            encoded = tokenizer.encode(value)
        except ValueError:
            continue
        if len(encoded) != 1 or encoded[0] in seen:
            continue
        ids.append(encoded[0])
        seen.add(encoded[0])
    return ids
