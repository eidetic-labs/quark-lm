"""Dispatch context-replay coverage modes for direct-answer training."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_context_replay_objective import (
    train_direct_answer_branch_context_replay_coverage_unlikelihood,
)
from transformer_direct_answer_core import DirectAnswerLesson


@dataclass(frozen=True)
class ContextReplayModeConfig:
    balance_targets: bool = False
    preserve_covered_targets: bool = False
    balance_covered_target_anchors: bool = False
    focus_uncovered_targets: bool = False
    preserve_predicted_target_coverage: bool = False
    balance_deficit_targets: bool = False
    profile_aware_targets: bool = False
    balance_profile_target_shares: bool = False
    enforce_prompt_target_margins: bool = False


_ANCHOR = ContextReplayModeConfig(preserve_covered_targets=True)
_BALANCED_ANCHOR = ContextReplayModeConfig(
    balance_targets=True,
    preserve_covered_targets=True,
)
_TARGET_BALANCED_ANCHOR = ContextReplayModeConfig(
    preserve_covered_targets=True,
    balance_covered_target_anchors=True,
)
_BALANCED_TARGET_BALANCED_ANCHOR = ContextReplayModeConfig(
    balance_targets=True,
    preserve_covered_targets=True,
    balance_covered_target_anchors=True,
)
_DEFICIT = ContextReplayModeConfig(focus_uncovered_targets=True)
_BALANCED_DEFICIT = ContextReplayModeConfig(
    balance_targets=True,
    focus_uncovered_targets=True,
)
_PRESERVING_DEFICIT = ContextReplayModeConfig(
    focus_uncovered_targets=True,
    preserve_predicted_target_coverage=True,
    balance_deficit_targets=True,
)
_BALANCED_PRESERVING_DEFICIT = ContextReplayModeConfig(
    balance_targets=True,
    focus_uncovered_targets=True,
    preserve_predicted_target_coverage=True,
    balance_deficit_targets=True,
)
_PROFILE_PRESERVING_DEFICIT = ContextReplayModeConfig(
    focus_uncovered_targets=True,
    preserve_predicted_target_coverage=True,
    balance_deficit_targets=True,
    profile_aware_targets=True,
)
_BALANCED_PROFILE_PRESERVING_DEFICIT = ContextReplayModeConfig(
    balance_targets=True,
    focus_uncovered_targets=True,
    preserve_predicted_target_coverage=True,
    balance_deficit_targets=True,
    profile_aware_targets=True,
)
_BALANCED_PROFILE_TARGET_SHARE_DEFICIT = ContextReplayModeConfig(
    balance_targets=True,
    focus_uncovered_targets=True,
    preserve_predicted_target_coverage=True,
    balance_deficit_targets=True,
    profile_aware_targets=True,
    balance_profile_target_shares=True,
)
_BALANCED_PROMPT_OWNERSHIP_TARGET_SHARE_DEFICIT = ContextReplayModeConfig(
    balance_targets=True,
    focus_uncovered_targets=True,
    preserve_predicted_target_coverage=True,
    balance_deficit_targets=True,
    profile_aware_targets=True,
    balance_profile_target_shares=True,
    enforce_prompt_target_margins=True,
)


CONTEXT_REPLAY_DIRECT_ANSWER_MODE_CONFIGS = {
    "branch-context-replay-coverage-unlikelihood": ContextReplayModeConfig(),
    "branch-balanced-context-replay-coverage-unlikelihood": ContextReplayModeConfig(
        balance_targets=True,
    ),
    "branch-context-coverage-anchor-unlikelihood": _ANCHOR,
    "branch-balanced-context-coverage-anchor-unlikelihood": _BALANCED_ANCHOR,
    "branch-context-target-balanced-anchor-unlikelihood": _TARGET_BALANCED_ANCHOR,
    "branch-balanced-context-target-balanced-anchor-unlikelihood": (
        _BALANCED_TARGET_BALANCED_ANCHOR
    ),
    "branch-context-coverage-deficit-unlikelihood": _DEFICIT,
    "branch-balanced-context-coverage-deficit-unlikelihood": _BALANCED_DEFICIT,
    "branch-context-coverage-preserving-deficit-unlikelihood": _PRESERVING_DEFICIT,
    "branch-balanced-context-coverage-preserving-deficit-unlikelihood": (
        _BALANCED_PRESERVING_DEFICIT
    ),
    "branch-context-profile-coverage-preserving-deficit-unlikelihood": (
        _PROFILE_PRESERVING_DEFICIT
    ),
    "branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood": (
        _BALANCED_PROFILE_PRESERVING_DEFICIT
    ),
    "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood": (
        _BALANCED_PROFILE_TARGET_SHARE_DEFICIT
    ),
    "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood": (
        _BALANCED_PROMPT_OWNERSHIP_TARGET_SHARE_DEFICIT
    ),
}
CONTEXT_REPLAY_DIRECT_ANSWER_MODES = frozenset(CONTEXT_REPLAY_DIRECT_ANSWER_MODE_CONFIGS)


def train_direct_answer_context_replay_mode_step(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    config = CONTEXT_REPLAY_DIRECT_ANSWER_MODE_CONFIGS.get(args.direct_answer_mode)
    if config is None:
        raise ValueError(
            f"Unsupported context-replay direct-answer mode: {args.direct_answer_mode}"
        )
    return train_direct_answer_branch_context_replay_coverage_unlikelihood(
        model,
        tokenizer,
        example,
        branch_examples,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        args.direct_answer_hard_negatives,
        terminator,
        params,
        balance_targets=config.balance_targets,
        preserve_covered_targets=config.preserve_covered_targets,
        balance_covered_target_anchors=config.balance_covered_target_anchors,
        focus_uncovered_targets=config.focus_uncovered_targets,
        preserve_predicted_target_coverage=config.preserve_predicted_target_coverage,
        balance_deficit_targets=config.balance_deficit_targets,
        profile_aware_targets=config.profile_aware_targets,
        balance_profile_target_shares=config.balance_profile_target_shares,
        enforce_prompt_target_margins=config.enforce_prompt_target_margins,
    )
