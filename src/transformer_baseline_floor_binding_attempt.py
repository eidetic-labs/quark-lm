"""Single-attempt evaluation for collapsed profile binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from replay_plan import BranchReplayRecord
from transformer_baseline_floor_profile_outcome_types import (
    BaselineFloorCollapsedProfileBindingOutcome,
)


@dataclass(frozen=True)
class CollapsedProfileBindingAttemptResult:
    loss: float
    floor_preserved: bool
    probe_snapshot: dict[str, Any]
    score: tuple[float, ...]
    delta: dict[str, Any]
    preservation_delta: dict[str, Any] | None
    outcome: str
    rejection_reason: str
    accepted: bool


def run_collapsed_profile_binding_attempt(
    *,
    model: Any,
    profile_batch: list[BranchReplayRecord],
    base_learning_rate: float,
    profile_scale: float,
    binding_learning_rate_scale: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    params: Any,
    direct_step: int,
    direct_baseline: dict[str, Any],
    snapshot_recorder: Any,
    update_guard: dict[str, Any],
    update_shape: str,
    profile: str,
    profile_frontier_records: int,
    target_profiles: list[str],
    profile_probe_snapshot: dict[str, Any],
    base_score: tuple[float, ...],
    memory_consolidation_active: bool,
    memory_consolidation_target_profiles: Sequence[str],
    owner_paraphrase_binding_active: bool,
    owner_paraphrase_preserved_profiles: Sequence[str],
    current_preservation_delta: dict[str, Any] | None,
    train_branch_diversity: Callable[..., float],
    preserves_target_coverage: Callable[[dict[str, Any], dict[str, Any]], bool],
    snapshot_score: Callable[[dict[str, Any]], tuple[float, ...]],
    target_coverage_delta: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ],
    profile_diversity_delta: Callable[
        [dict[str, Any], dict[str, Any], Sequence[str]], dict[str, Any]
    ],
    evaluate_outcome: Callable[
        ..., BaselineFloorCollapsedProfileBindingOutcome
    ],
) -> CollapsedProfileBindingAttemptResult:
    loss = train_branch_diversity(
        model,
        profile_batch,
        base_learning_rate * profile_scale * binding_learning_rate_scale,
        negative_weight,
        positive_weight,
        contrast_weight,
        params=params,
    )
    probe_snapshot = snapshot_recorder.record(
        direct_step,
        None,
        _binding_probe_metadata(
            profile=profile,
            profile_batch=profile_batch,
            profile_scale=profile_scale,
            binding_learning_rate_scale=binding_learning_rate_scale,
            update_shape=update_shape,
            profile_frontier_records=profile_frontier_records,
            target_profiles=target_profiles,
            memory_consolidation_active=memory_consolidation_active,
            memory_consolidation_target_profiles=memory_consolidation_target_profiles,
            owner_paraphrase_binding_active=owner_paraphrase_binding_active,
            owner_paraphrase_preserved_profiles=owner_paraphrase_preserved_profiles,
        ),
    )
    floor_preserved = preserves_target_coverage(probe_snapshot, direct_baseline)
    score = snapshot_score(probe_snapshot)
    coverage_delta = target_coverage_delta(probe_snapshot, profile_probe_snapshot)
    collapsed_delta = profile_diversity_delta(
        probe_snapshot,
        profile_probe_snapshot,
        target_profiles,
    )
    owner_preservation_regressed = False
    if owner_paraphrase_binding_active:
        update_guard["profile_scale_owner_paraphrase_binding_preservation_checks"] += 1
        current_preservation_delta = profile_diversity_delta(
            probe_snapshot,
            profile_probe_snapshot,
            owner_paraphrase_preserved_profiles,
        )
        owner_preservation_regressed = (
            int(current_preservation_delta["regressed_profile_count"]) > 0
        )
    delta = {
        "coverage_delta": coverage_delta,
        "profile_delta": collapsed_delta,
    }
    if current_preservation_delta is not None:
        delta["owner_paraphrase_preservation_delta"] = current_preservation_delta
    outcome = evaluate_outcome(
        floor_preserved=floor_preserved,
        binding_score=score,
        base_score=base_score,
        coverage_delta=coverage_delta,
        profile_delta=collapsed_delta,
        owner_paraphrase_preservation_regressed=owner_preservation_regressed,
    )
    if outcome.owner_paraphrase_preservation_failed:
        update_guard["profile_scale_owner_paraphrase_binding_preservation_failures"] += 1
    return CollapsedProfileBindingAttemptResult(
        loss=loss,
        floor_preserved=floor_preserved,
        probe_snapshot=probe_snapshot,
        score=score,
        delta=delta,
        preservation_delta=current_preservation_delta,
        outcome=outcome.outcome,
        rejection_reason=outcome.rejection_reason,
        accepted=outcome.accepted,
    )


def _binding_probe_metadata(
    *,
    profile: str,
    profile_batch: list[BranchReplayRecord],
    profile_scale: float,
    binding_learning_rate_scale: float,
    update_shape: str,
    profile_frontier_records: int,
    target_profiles: list[str],
    memory_consolidation_active: bool,
    memory_consolidation_target_profiles: Sequence[str],
    owner_paraphrase_binding_active: bool,
    owner_paraphrase_preserved_profiles: Sequence[str],
) -> dict[str, Any]:
    return {
        "baseline_floor_update_guard_probe": True,
        "baseline_floor_sequential_profile_probe": True,
        "baseline_floor_calibrated_sequential_profile_probe": True,
        "baseline_floor_profile_scale_memory_probe": True,
        "baseline_floor_profile_scale_frontier_probe": True,
        "baseline_floor_profile_scale_collapsed_profile_binding_probe": True,
        "learning_rate_scale": profile_scale,
        "collapsed_profile_binding_learning_rate_scale": binding_learning_rate_scale,
        "update_shape": update_shape,
        "sequential_profile": profile,
        "sequential_profile_records": len(profile_batch),
        "sequential_profile_frontier_records": profile_frontier_records,
        "collapsed_profile_binding_records": len(profile_batch),
        "collapsed_profile_binding_target_profiles": target_profiles,
        "memory_consolidation_target_profiles": (
            list(memory_consolidation_target_profiles)
            if memory_consolidation_active
            else []
        ),
        "owner_paraphrase_binding_preserved_profiles": (
            list(owner_paraphrase_preserved_profiles)
            if owner_paraphrase_binding_active
            else []
        ),
    }
