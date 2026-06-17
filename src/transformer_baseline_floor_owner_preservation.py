"""Owner-paraphrase preservation checks for profile-scale baseline-floor updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from branch_diversity_snapshots import (
    branch_diversity_snapshot_profile_diversity_delta,
)


@dataclass(frozen=True)
class OwnerParaphraseBindingPreservation:
    preserved: bool
    delta: dict[str, Any] | None = None
    rejection_reason: str = ""


def check_owner_paraphrase_binding_preservation(
    *,
    active: bool,
    update_guard: dict[str, Any],
    profile_probe_snapshot: dict[str, Any] | None,
    profile_base_snapshot: dict[str, Any] | None,
    preserved_profiles: Sequence[str],
    profile_diversity_delta: Callable[
        [dict[str, Any], dict[str, Any], Sequence[str]], dict[str, Any]
    ] = branch_diversity_snapshot_profile_diversity_delta,
) -> OwnerParaphraseBindingPreservation:
    if not (active and profile_probe_snapshot is not None and profile_base_snapshot):
        return OwnerParaphraseBindingPreservation(preserved=True)

    update_guard["profile_scale_owner_paraphrase_binding_preservation_checks"] += 1
    delta = profile_diversity_delta(
        profile_probe_snapshot,
        profile_base_snapshot,
        preserved_profiles,
    )
    if int(delta["regressed_profile_count"]) <= 0:
        return OwnerParaphraseBindingPreservation(preserved=True, delta=delta)

    update_guard["profile_scale_owner_paraphrase_binding_preservation_failures"] += 1
    return OwnerParaphraseBindingPreservation(
        preserved=False,
        delta=delta,
        rejection_reason="owner_paraphrase_preservation_regression",
    )
