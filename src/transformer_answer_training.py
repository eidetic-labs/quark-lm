"""High-level transformer answer-training workflow."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from tokenizer import CharTokenizer
from transformer_answer_core_stage import train_core_answer_stage
from transformer_answer_direct_stage import run_transformer_direct_answer_stage
from transformer_answer_run_completion import complete_transformer_answer_training_run
from transformer_answer_run_setup import prepare_transformer_answer_run
from transformer_answer_training_snapshots import build_answer_training_snapshot_callback


def train_transformer_answers_command(
    args: argparse.Namespace,
    model_class: type[Any],
    initialize_transformer_for_training_fn: Callable[
        [argparse.Namespace, CharTokenizer],
        tuple[Any, dict[str, Any]],
    ],
) -> dict[str, Any]:
    setup = prepare_transformer_answer_run(
        args,
        initialize_transformer_for_training_fn,
        model_class,
    )
    model = setup.model
    tokenizer = setup.tokenizer
    optimizer = setup.optimizer
    training_plan = setup.training_plan

    snapshot = build_answer_training_snapshot_callback(
        args=args,
        setup=setup,
        model=lambda: model,
        tokenizer=lambda: tokenizer,
    )
    baseline, last_snapshot = train_core_answer_stage(
        args,
        model,
        tokenizer,
        setup.training_pool,
        setup.rng,
        setup.training_candidates,
        snapshot,
    )

    post_direct_candidate_snapshot: dict[str, Any] | None = None
    post_direct_candidate_snapshot_skipped = False
    direct_answer_metrics: dict[str, Any] | None = None
    if args.direct_answer_steps > 0:
        direct_result = run_transformer_direct_answer_stage(
            args=args,
            model_class=model_class,
            setup=setup,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            training_plan=training_plan,
            last_snapshot=last_snapshot,
            snapshot=snapshot,
        )
        model = direct_result.model
        tokenizer = direct_result.tokenizer
        optimizer = direct_result.optimizer
        training_plan = direct_result.training_plan
        last_snapshot = direct_result.last_snapshot
        post_direct_candidate_snapshot = (
            direct_result.post_direct_candidate_snapshot
        )
        post_direct_candidate_snapshot_skipped = (
            direct_result.post_direct_candidate_snapshot_skipped
        )
        direct_answer_metrics = direct_result.direct_answer_metrics

    return complete_transformer_answer_training_run(
        args=args,
        setup=setup,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        training_plan=training_plan,
        baseline=baseline,
        last_snapshot=last_snapshot,
        post_direct_candidate_snapshot=post_direct_candidate_snapshot,
        post_direct_candidate_snapshot_skipped=post_direct_candidate_snapshot_skipped,
        direct_answer_metrics=direct_answer_metrics,
    )
