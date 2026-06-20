"""Declared direct-answer repair target profiles."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def direct_answer_repair_target_profiles(args: Any) -> list[str]:
    return normalized_repair_target_profiles(
        getattr(args, "direct_answer_repair_target_profile", None)
    )


def normalized_repair_target_profiles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values: Iterable[Any] = [value]
    else:
        raw_values = value if isinstance(value, Iterable) else [value]
    profiles = {
        str(profile).strip()
        for profile in raw_values
        if str(profile).strip()
    }
    return sorted(profiles)
