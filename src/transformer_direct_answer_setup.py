"""Direct-answer training setup and replay-plan preparation."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answer_model import AnswerExample
from replay_plan import BranchReplayRecord
from tokenizer import CharTokenizer
from transformer_answer_training_helpers import (
    normalize_answer_terminator,
    transformer_direct_answer_training_pool,
)
from transformer_direct_answer_core import DirectAnswerLesson, direct_answer_lesson
from transformer_direct_answer_mode_flags import direct_answer_mode_flags
from transformer_direct_answer_memory_setup import (
    direct_memory_field_kwargs,
    prepare_direct_memory_consolidation,
    remaining_profile_binding_targets,
)
from transformer_direct_answer_replay_setup import (
    prepare_direct_answer_replay_plan,
)
from transformer_direct_answer_replay_artifacts import write_direct_answer_replay_plan
from transformer_direct_answer_setup_flags import direct_answer_setup_flag_field_kwargs
import transformer_direct_modes as modes
from transformer_experiment import is_profile_aware_direct_answer_mode
from transformer_memory_plan_helpers import (
    remaining_profile_binding_source_labels,
)
from transformer_training import JsonlHistoryWriter


@dataclass
class DirectAnswerRunSetup:
    direct_answer_terminator: str
    direct_training_pool: list[AnswerExample]
    direct_lessons: dict[AnswerExample, DirectAnswerLesson]
    direct_rng: random.Random
    direct_history_path: Path
    direct_history_writer: JsonlHistoryWriter
    direct_profile_aware_targets: bool
    direct_replay_plan_path: Path | None
    direct_replay_plan: dict[str, Any] | None
    direct_replay_prediction_overrides: modes.ReplayPredictionOverrides | None
    direct_replay_prediction_anchors_active: bool
    direct_answer_baseline_floor_update_gate_active: bool
    direct_answer_baseline_floor_adaptive_updates_active: bool
    direct_answer_baseline_floor_repaired_updates_active: bool
    direct_answer_baseline_floor_objective_active: bool
    direct_answer_baseline_floor_stabilization_active: bool
    direct_answer_baseline_floor_profile_targeted_stabilization_active: bool
    direct_answer_baseline_floor_sequential_stabilization_active: bool
    direct_answer_baseline_floor_calibrated_sequential_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_diversity_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_missing_first_token_consolidation_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_remaining_collapsed_missing_first_token_consolidation_frontier_stabilization_active: bool
    direct_answer_baseline_floor_profile_scale_remaining_collapsed_profile_specific_missing_first_token_consolidation_frontier_stabilization_active: bool
    direct_memory_consolidation_source_plan_path: Path | None
    direct_memory_consolidation_source_plan_summary: dict[str, Any]
    direct_memory_consolidation_target_profiles: list[str]
    direct_memory_consolidation_top_priority_profiles: list[str]
    direct_memory_consolidation_collapsed_memory_backed_profiles: list[str]
    direct_memory_consolidation_missing_first_token_values: dict[str, list[str]]
    direct_memory_consolidation_missing_first_token_ids: dict[str, list[int]]
    direct_memory_consolidation_profile_specific_missing_first_token_target_map: dict[
        str, list[str]
    ]
    direct_remaining_profile_binding_target_profiles: list[str]
    direct_remaining_profile_binding_source_labels: list[str]
    direct_baseline_floor_learning_rate_scales: tuple[float, ...]
    direct_baseline_floor_outer_learning_rate_scales: tuple[float, ...]
    direct_replay_records: list[BranchReplayRecord]
    direct_baseline_floor_repair_anchors: list[BranchReplayRecord]
    direct_baseline_floor_frontier_anchors: list[BranchReplayRecord]
    training_plan: dict[str, Any]


def prepare_direct_answer_run_setup(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    examples: list[AnswerExample],
    training_plan: dict[str, Any],
    training_plan_path: Path,
) -> DirectAnswerRunSetup:
    terminator = normalize_answer_terminator(args.direct_answer_terminator)
    if terminator and terminator not in tokenizer.stoi:
        raise ValueError(
            "direct answer terminator is outside the admitted tokenizer vocabulary"
        )
    training_pool = transformer_direct_answer_training_pool(examples)
    lessons = _direct_lessons(
        tokenizer,
        model.config.context_size,
        training_pool,
        terminator,
    )
    flags = direct_answer_mode_flags(args.direct_answer_mode)
    memory = prepare_direct_memory_consolidation(args, tokenizer, flags)
    remaining_targets = remaining_profile_binding_targets(memory, flags)
    remaining_source_labels = (
        remaining_profile_binding_source_labels(remaining_targets)
        if flags["profile_scale_remaining_profile_binding_frontier_stabilization_active"]
        else []
    )
    learning_rate_scales = (
        modes.BASELINE_FLOOR_CALIBRATED_ADAPTIVE_LEARNING_RATE_SCALES
        if flags["calibrated_sequential_stabilization_active"]
        else modes.BASELINE_FLOOR_ADAPTIVE_LEARNING_RATE_SCALES
    )
    outer_learning_rate_scales = (
        (1.0,)
        if flags["profile_scale_calibrated_stabilization_active"]
        else learning_rate_scales
    )
    profile_aware_targets = is_profile_aware_direct_answer_mode(args.direct_answer_mode)
    replay_setup = prepare_direct_answer_replay_plan(
        args,
        model,
        tokenizer,
        training_pool,
        terminator,
        profile_aware_targets,
        flags,
        memory,
        remaining_targets,
        remaining_source_labels,
        learning_rate_scales,
        outer_learning_rate_scales,
    )
    training_plan = write_direct_answer_replay_plan(
        training_plan,
        training_plan_path,
        replay_setup["direct_replay_plan"],
        replay_setup["direct_replay_plan_path"],
    )
    return DirectAnswerRunSetup(
        direct_answer_terminator=terminator,
        direct_training_pool=training_pool,
        direct_lessons=lessons,
        direct_rng=random.Random(args.seed + 307),
        direct_history_path=args.run / "direct_answer_metrics.jsonl",
        direct_history_writer=JsonlHistoryWriter(args.run / "direct_answer_metrics.jsonl"),
        direct_profile_aware_targets=profile_aware_targets,
        direct_replay_prediction_anchors_active=(
            args.direct_answer_mode in modes.BASELINE_ANCHORED_DIRECT_ANSWER_MODES
        ),
        **direct_answer_setup_flag_field_kwargs(flags),
        **direct_memory_field_kwargs(memory),
        direct_remaining_profile_binding_target_profiles=remaining_targets,
        direct_remaining_profile_binding_source_labels=remaining_source_labels,
        direct_baseline_floor_learning_rate_scales=tuple(learning_rate_scales),
        direct_baseline_floor_outer_learning_rate_scales=tuple(outer_learning_rate_scales),
        training_plan=training_plan,
        **replay_setup,
    )


def _direct_lessons(
    tokenizer: CharTokenizer,
    context_size: int,
    training_pool: list[AnswerExample],
    terminator: str,
) -> dict[AnswerExample, DirectAnswerLesson]:
    return {
        example: direct_answer_lesson(tokenizer, context_size, example, terminator)
        for example in sorted(
            set(training_pool),
            key=lambda item: (item.prompt, item.target, item.source),
        )
    }
