"""Direct-answer replay-plan preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from answer_model import AnswerExample
from replay_plan import (
    BranchReplayRecord,
    branch_replay_parts,
    branch_replay_plan,
)
from tokenizer import CharTokenizer
from transformer_baseline_floor_anchor_selection import (
    baseline_floor_frontier_anchor_records,
    baseline_floor_repair_anchor_records,
)
from transformer_direct_answer_batches import direct_answer_profiled_replay_records
from transformer_direct_answer_replay_artifacts import write_direct_answer_replay_plan
from transformer_direct_answer_replay_summary import attach_direct_answer_replay_summary


def prepare_direct_answer_replay_plan(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    training_pool: list[AnswerExample],
    terminator: str,
    profile_aware_targets: bool,
    flags: dict[str, bool],
    memory: Any,
    remaining_targets: list[str],
    remaining_source_labels: list[str],
    learning_rate_scales: tuple[float, ...],
    outer_learning_rate_scales: tuple[float, ...],
) -> dict[str, Any]:
    replay_plan_path = (
        args.run / "direct_answer_replay_plan.json" if profile_aware_targets else None
    )
    if not profile_aware_targets:
        return _empty_replay_setup(replay_plan_path)
    replay_records = direct_answer_profiled_replay_records(
        model,
        tokenizer,
        training_pool,
        args.direct_answer_branch_position,
        terminator,
    )
    repair_anchors = (
        baseline_floor_repair_anchor_records(replay_records)
        if (
            flags["repair_active"]
            or flags["objective_active"]
            or flags["stabilization_active"]
        )
        else []
    )
    frontier_anchors = (
        baseline_floor_frontier_anchor_records(repair_anchors, replay_records)
        if flags["profile_scale_frontier_stabilization_active"]
        else []
    )
    prediction_overrides = {
        (tuple(context), target, profile): predicted
        for context, target, predicted, profile in (
            branch_replay_parts(record) for record in replay_records
        )
    }
    replay_plan = branch_replay_plan(
        replay_records,
        replay_records,
        profile_aware_targets=True,
    )
    attach_direct_answer_replay_summary(
        replay_plan,
        args,
        training_pool,
        flags,
        memory,
        remaining_targets,
        remaining_source_labels,
        learning_rate_scales,
        outer_learning_rate_scales,
        prediction_overrides,
        repair_anchors,
        frontier_anchors,
    )
    return {
        "direct_replay_plan_path": replay_plan_path,
        "direct_replay_plan": replay_plan,
        "direct_replay_prediction_overrides": prediction_overrides,
        "direct_replay_records": replay_records,
        "direct_baseline_floor_repair_anchors": repair_anchors,
        "direct_baseline_floor_frontier_anchors": frontier_anchors,
    }


def _empty_replay_setup(replay_plan_path: Path | None) -> dict[str, Any]:
    return {
        "direct_replay_plan_path": replay_plan_path,
        "direct_replay_plan": None,
        "direct_replay_prediction_overrides": None,
        "direct_replay_records": [],
        "direct_baseline_floor_repair_anchors": [],
        "direct_baseline_floor_frontier_anchors": [],
    }
