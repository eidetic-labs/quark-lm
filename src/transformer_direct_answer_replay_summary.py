"""Summary decoration for direct-answer replay plans."""

from __future__ import annotations

import argparse
from typing import Any

from answer_model import AnswerExample
from replay_plan import BranchReplayRecord
from transformer_direct_answer_replay_anchor_summary import attach_anchor_summary
from transformer_direct_answer_replay_binding_summary import attach_binding_summary
from transformer_direct_answer_replay_flags import replay_flag_summary_fields
import transformer_direct_modes as modes
from transformer_direct_modes import ReplayPredictionOverrides


def attach_direct_answer_replay_summary(
    replay_plan: dict[str, Any],
    args: argparse.Namespace,
    training_pool: list[AnswerExample],
    flags: dict[str, bool],
    memory: Any,
    remaining_targets: list[str],
    remaining_source_labels: list[str],
    learning_rate_scales: tuple[float, ...],
    outer_learning_rate_scales: tuple[float, ...],
    prediction_overrides: ReplayPredictionOverrides,
    repair_anchors: list[BranchReplayRecord],
    frontier_anchors: list[BranchReplayRecord],
) -> None:
    replay_plan.update(
        {
            "mode": args.direct_answer_mode,
            "branch_position": args.direct_answer_branch_position,
            "training_examples": len(training_pool),
            "baseline_prediction_anchor_count": len(prediction_overrides),
            "baseline_prediction_anchors_active": (
                args.direct_answer_mode in modes.BASELINE_ANCHORED_DIRECT_ANSWER_MODES
            ),
            **replay_flag_summary_fields(flags),
            "baseline_floor_repair_anchor_count": len(repair_anchors),
            "baseline_floor_repair_steps": (
                modes.BASELINE_FLOOR_REPAIR_STEPS if flags["repair_active"] else 0
            ),
            "baseline_floor_objective_anchor_count": len(repair_anchors),
            "baseline_floor_objective_anchor_batch_size": (
                modes.BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE
                if flags["objective_active"]
                else 0
            ),
            "baseline_floor_objective_anchor_weight": (
                modes.BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT
                if flags["objective_active"]
                else 0.0
            ),
        }
    )
    attach_anchor_summary(replay_plan, repair_anchors, frontier_anchors, flags)
    if flags["adaptive"]:
        replay_plan["adaptive_learning_rate_scales"] = list(learning_rate_scales)
        replay_plan["outer_learning_rate_scales"] = list(outer_learning_rate_scales)
    if flags["profile_scale_collapsed_profile_binding_frontier_stabilization_active"]:
        replay_plan["collapsed_profile_binding_learning_rate_scales"] = list(
            modes.BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES
        )
    attach_binding_summary(
        replay_plan,
        args,
        flags,
        memory,
        remaining_targets,
        remaining_source_labels,
    )
