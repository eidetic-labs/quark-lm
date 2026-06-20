"""Dispatch one direct-answer training step across objective families."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_branch_basic_dispatch import (
    BASIC_BRANCH_DIRECT_ANSWER_MODES,
    train_direct_answer_branch_basic_mode_step,
)
from transformer_direct_answer_branch_binding_dispatch import (
    BRANCH_BINDING_DIRECT_ANSWER_MODES,
    train_direct_answer_branch_binding_mode_step,
)
from transformer_direct_answer_branch_contrast_dispatch import (
    BRANCH_CONTRAST_DIRECT_ANSWER_MODES,
    train_direct_answer_branch_contrast_mode_step,
)
from transformer_direct_answer_context_replay_dispatch import (
    CONTEXT_REPLAY_DIRECT_ANSWER_MODES,
    train_direct_answer_context_replay_mode_step,
)
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_error_dispatch import (
    train_direct_answer_error_repair_mode_step,
)
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import BASELINE_ANCHORED_DIRECT_ANSWER_MODES


@dataclass(frozen=True)
class DirectAnswerModeStepResult:
    loss: float
    update_guard_applied: bool = False


AdaptiveBaselineFloorUpdate = Callable[
    [int, dict[str, Any], dict[str, Any], object],
    float,
]
BaselineAnchoredPromptUpdate = Callable[[AnswerExample, DirectAnswerLesson, float], float]


def train_direct_answer_mode_step(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    direct_step: int,
    terminator: str,
    params: list[Scalar],
    baseline_floor_adaptive_updates_active: bool,
    pre_update_model_payload: dict[str, Any] | None,
    pre_update_optimizer_payload: dict[str, Any] | None,
    pre_update_rng_state: object | None,
    train_adaptive_baseline_floor_update: AdaptiveBaselineFloorUpdate,
    train_baseline_anchored_prompt: BaselineAnchoredPromptUpdate,
    eval_records: dict[str, list[dict[str, Any]]] | None = None,
) -> DirectAnswerModeStepResult:
    error_repair_loss = train_direct_answer_error_repair_mode_step(
        args=args,
        model=model,
        tokenizer=tokenizer,
        example=example,
        lesson=lesson,
        rng=rng,
        direct_step=direct_step,
        terminator=terminator,
        params=params,
    )
    if error_repair_loss is not None:
        return DirectAnswerModeStepResult(error_repair_loss)
    if args.direct_answer_mode in BASIC_BRANCH_DIRECT_ANSWER_MODES:
        return DirectAnswerModeStepResult(
            train_direct_answer_branch_basic_mode_step(
                args=args,
                model=model,
                tokenizer=tokenizer,
                example=example,
                lesson=lesson,
                branch_examples=branch_examples,
                eval_records=eval_records,
                rng=rng,
                direct_step=direct_step,
                terminator=terminator,
                params=params,
            )
        )
    if args.direct_answer_mode in BRANCH_BINDING_DIRECT_ANSWER_MODES:
        return DirectAnswerModeStepResult(
            train_direct_answer_branch_binding_mode_step(
                args=args,
                model=model,
                tokenizer=tokenizer,
                example=example,
                lesson=lesson,
                branch_examples=branch_examples,
                rng=rng,
                terminator=terminator,
                params=params,
            )
        )
    if args.direct_answer_mode in CONTEXT_REPLAY_DIRECT_ANSWER_MODES:
        return DirectAnswerModeStepResult(
            train_direct_answer_context_replay_mode_step(
                args=args,
                model=model,
                tokenizer=tokenizer,
                example=example,
                lesson=lesson,
                branch_examples=branch_examples,
                rng=rng,
                terminator=terminator,
                params=params,
            )
        )
    if args.direct_answer_mode in BASELINE_ANCHORED_DIRECT_ANSWER_MODES:
        if (
            baseline_floor_adaptive_updates_active
            and pre_update_model_payload is not None
            and pre_update_optimizer_payload is not None
            and pre_update_rng_state is not None
        ):
            return DirectAnswerModeStepResult(
                train_adaptive_baseline_floor_update(
                    direct_step,
                    pre_update_model_payload,
                    pre_update_optimizer_payload,
                    pre_update_rng_state,
                ),
                update_guard_applied=True,
            )
        return DirectAnswerModeStepResult(
            train_baseline_anchored_prompt(
                example,
                lesson,
                args.direct_answer_learning_rate,
            )
        )
    if args.direct_answer_mode in BRANCH_CONTRAST_DIRECT_ANSWER_MODES:
        return DirectAnswerModeStepResult(
            train_direct_answer_branch_contrast_mode_step(
                args=args,
                model=model,
                tokenizer=tokenizer,
                example=example,
                lesson=lesson,
                branch_examples=branch_examples,
                rng=rng,
                direct_step=direct_step,
                terminator=terminator,
                params=params,
            )
        )
    return DirectAnswerModeStepResult(
        train_direct_answer_lesson(
            model,
            lesson,
            rng,
            args.direct_answer_learning_rate,
            params,
        )
    )
