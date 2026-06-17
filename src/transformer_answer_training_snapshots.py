"""Snapshot records for answer, selector, and generator training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tokenizer import CharTokenizer
from transformer_answer_evaluation import evaluate_answer_records
from transformer_answer_generator import (
    TransformerGuidedAnswerGenerator,
    evaluate_answer_generator_records,
)
from transformer_answer_selector import AnswerCandidateSelector
from transformer_model import GenerationConfig


def answer_training_snapshot_record(
    model: Any,
    tokenizer: CharTokenizer,
    eval_records: dict[str, list[dict[str, Any]]],
    eval_candidates: dict[str, list[str]],
    candidates: list[str],
    candidate_scope: str,
    max_new_chars: int,
    include_completions: bool,
    generation_config: GenerationConfig,
    step: int,
    train_loss: float | None,
    train_target_loss: float | None = None,
    train_choice_loss: float | None = None,
    train_choice_candidates: float | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "train_loss": train_loss,
        "train_target_loss": train_target_loss,
        "train_choice_loss": train_choice_loss,
        "train_choice_candidates": train_choice_candidates,
        "evals": {
            name: evaluate_answer_records(
                model,
                tokenizer,
                records,
                candidates if candidate_scope == "all" else eval_candidates[name],
                max_new_chars,
                include_completions=include_completions,
                generation_config=generation_config,
            )
            for name, records in sorted(eval_records.items())
        },
    }


def build_answer_training_snapshot_callback(
    *,
    args: Any,
    setup: Any,
    model: Callable[[], Any],
    tokenizer: Callable[[], CharTokenizer],
) -> Callable[[int, float | None, float | None, float | None, float | None], dict[str, Any]]:
    def snapshot(
        step: int,
        train_loss: float | None,
        train_target_loss: float | None = None,
        train_choice_loss: float | None = None,
        train_choice_candidates: float | None = None,
    ) -> dict[str, Any]:
        return setup.history_writer.append(
            answer_training_snapshot_record(
                model(),
                tokenizer(),
                setup.eval_records,
                setup.eval_candidates,
                setup.candidates,
                args.candidate_scope,
                args.max_new_chars,
                args.include_completions,
                setup.generation_config,
                step,
                train_loss,
                train_target_loss,
                train_choice_loss,
                train_choice_candidates,
            )
        )

    return snapshot


def selector_snapshot_record(
    model: Any,
    tokenizer: CharTokenizer,
    selector: AnswerCandidateSelector,
    eval_records: dict[str, list[dict[str, Any]]],
    eval_candidates: dict[str, list[str]],
    candidates: list[str],
    candidate_scope: str,
    max_new_chars: int,
    emit_selected_candidate: bool,
    step: int,
    train_loss: float | None,
) -> dict[str, Any]:
    return {
        "step": step,
        "train_loss": train_loss,
        "evals": {
            name: evaluate_answer_records(
                model,
                tokenizer,
                records,
                candidates if candidate_scope == "all" else eval_candidates[name],
                max_new_chars,
                include_completions=False,
                selector=selector,
                emit_selected_candidate=emit_selected_candidate,
            )
            for name, records in sorted(eval_records.items())
        },
    }


def generator_snapshot_record(
    generator: TransformerGuidedAnswerGenerator,
    model: Any,
    tokenizer: CharTokenizer,
    eval_records: dict[str, list[dict[str, Any]]],
    step: int,
    train_loss: float | None,
) -> dict[str, Any]:
    return {
        "step": step,
        "train_loss": train_loss,
        "evals": {
            name: evaluate_answer_generator_records(
                generator,
                model,
                tokenizer,
                records,
            )
            for name, records in sorted(eval_records.items())
        },
    }
