"""Snapshot field extraction for branch frontier profile diagnostics."""

from __future__ import annotations

from typing import Any


def snapshot_branch_profile(snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    return _as_dict(_as_dict(snapshot.get("branch_profiles")).get(name))


def snapshot_representation_profile(
    snapshot: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    return _as_dict(_as_dict(snapshot.get("branch_representation_profiles")).get(name))


def snapshot_logit_prior_profile(
    snapshot: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    return _as_dict(_as_dict(snapshot.get("branch_logit_prior_profiles")).get(name))


def logit_prior_summary(profile: dict[str, Any]) -> dict[str, Any]:
    decomposition = _as_dict(profile.get("dominant_vs_target_decomposition"))
    failed = _as_dict(decomposition.get("failed_records"))
    if int(failed.get("count", 0)):
        return failed
    return _as_dict(decomposition.get("all_records"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
