"""Batch evidence for profile-balanced routing-repair bundles."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from replay_plan import branch_replay_parts
from transformer_direct_answer_profile_balanced_batches import (
    direct_answer_profile_balanced_branch_batch,
)
from transformer_direct_answer_profile_keys import trainable_eval_profile_keys
from transformer_routing_repair_bundle import (
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE,
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_MODE,
    PROFILE_BALANCED_ROUTING_REPAIR_MODE,
    routing_repair_bundle_supports_mode,
)

ROUTING_REPAIR_BATCH_MODE = PROFILE_BALANCED_ROUTING_REPAIR_MODE
ROUTING_REPAIR_RANK_BATCH_MODE = PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE
ROUTING_REPAIR_TOPK_BATCH_MODE = PROFILE_BALANCED_TOPK_ROUTING_REPAIR_MODE
ROUTING_REPAIR_BATCH_MODES = (
    ROUTING_REPAIR_BATCH_MODE,
    ROUTING_REPAIR_RANK_BATCH_MODE,
    ROUTING_REPAIR_TOPK_BATCH_MODE,
)


def routing_repair_batch_evidence_enabled(args: Any) -> bool:
    return routing_repair_bundle_supports_mode(
        getattr(args, "experiment_bundle", None),
        getattr(args, "direct_answer_mode", None),
    )


def record_routing_repair_batch_step(
    *,
    args: Any,
    model: Any,
    tokenizer: Any,
    branch_examples: list[Any],
    rng: random.Random,
    direct_step: int,
    terminator: str,
) -> dict[str, Any] | None:
    """Record the same profile-balanced branch surface used by Bundle A."""

    if not routing_repair_batch_evidence_enabled(args):
        return None
    probe_rng = random.Random()
    probe_rng.setstate(rng.getstate())
    branches = direct_answer_profile_balanced_branch_batch(
        model,
        tokenizer,
        branch_examples,
        probe_rng,
        getattr(args, "direct_answer_branch_position", 1),
        getattr(args, "direct_answer_branch_batch_size", 1),
        terminator,
    )
    return _step_record(direct_step, branches)


def routing_repair_batch_evidence_summary(
    args: Any,
    step_records: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any] | None:
    """Summarize whether Bundle A covered trainable failed profile families."""

    if not routing_repair_batch_evidence_enabled(args):
        return None
    required = _required_eval_profiles(baseline)
    trainable = trainable_eval_profile_keys()
    trainable_required = sorted(set(required["failed_profiles"]) & trainable)
    covered_profiles = sorted(
        {
            profile
            for record in step_records
            for profile in record.get("profiles", [])
        }
    )
    covered_trainable = sorted(set(covered_profiles) & set(trainable_required))
    missing_trainable = sorted(set(trainable_required) - set(covered_trainable))
    unmapped_eval_only = sorted(set(required["failed_profiles"]) - trainable)
    batch_check = {
        "name": "profile_balanced_branch_batches",
        "passed": bool(step_records) and not missing_trainable,
        "status": "passed"
        if bool(step_records) and not missing_trainable
        else "failed",
        "required_trainable_profiles": trainable_required,
        "covered_trainable_profiles": covered_trainable,
        "missing_trainable_profiles": missing_trainable,
        "eval_only_profiles": unmapped_eval_only,
    }
    return {
        "bundle": getattr(args, "experiment_bundle", None),
        "direct_answer_mode": getattr(args, "direct_answer_mode", None),
        "batch_builder": "profile-balanced-training-family-branch-batch",
        "step_count": len(step_records),
        "branch_count": sum(
            int(record.get("branch_count", 0)) for record in step_records
        ),
        "profiles": covered_profiles,
        "required_eval_profiles": required,
        "profile_balanced_branch_batches": batch_check,
        "steps": step_records,
        "passed": batch_check["passed"],
        "status": batch_check["status"],
    }


def _step_record(
    direct_step: int,
    branches: list[Any],
) -> dict[str, Any]:
    profiles: Counter[str] = Counter()
    targets: Counter[int] = Counter()
    predictions: Counter[int] = Counter()
    represented_targets: Counter[int] = Counter()
    for branch in branches:
        _context, target, predicted, profile = branch_replay_parts(branch)
        profiles[profile] += 1
        targets[target] += 1
        predictions[predicted] += 1
        if predicted == target:
            represented_targets[target] += 1
    return {
        "step": direct_step,
        "branch_count": len(branches),
        "profiles": sorted(profiles),
        "profile_counts": dict(sorted(profiles.items())),
        "target_count": len(targets),
        "predicted_count": len(predictions),
        "represented_target_count": len(represented_targets),
    }


def _required_eval_profiles(baseline: dict[str, Any]) -> dict[str, Any]:
    root_cause = (
        baseline.get("branch_diversity_target", {})
        .get("root_cause", {})
        if isinstance(baseline, dict)
        else {}
    )
    profiles = root_cause.get("profiles", []) if isinstance(root_cause, dict) else []
    failed = []
    zero_coverage = []
    buried = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        name = str(profile.get("name", ""))
        if not name:
            continue
        failed.append(name)
        modes = set(profile.get("failure_modes", []))
        if "zero_target_coverage" in modes:
            zero_coverage.append(name)
        if "targets_buried" in modes:
            buried.append(name)
    return {
        "failed_profiles": sorted(failed),
        "zero_coverage_profiles": sorted(zero_coverage),
        "buried_target_profiles": sorted(buried),
    }
