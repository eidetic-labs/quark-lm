"""Training-recipe construction for transformer answer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from answer_model import DEFAULT_EVALS as DEFAULT_ANSWER_EVALS
from training_recipe_core import build_training_recipe
from transformer_experiment_constants import (
    TRAINING_DATA_DESCRIPTION,
    TRANSFORMER_RECIPE_VERSION,
)
from transformer_experiment_modes import is_profile_aware_direct_answer_mode
from transformer_model import TRANSFORMER_ARCHITECTURE, TRANSFORMER_TOKENIZER


def transformer_training_recipe_id(args: Any) -> str:
    mode = (
        getattr(args, "direct_answer_mode", "target-loss")
        if getattr(args, "direct_answer_steps", 0) > 0
        else "target-loss"
    )
    return f"transformer-answer:{mode}:{TRANSFORMER_RECIPE_VERSION}"


def transformer_training_recipe(
    args: Any,
    tokenizer: Any,
    planned_artifacts: list[Path],
    acceptance_gates: list[dict[str, Any]],
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    generation_config: dict[str, Any],
    replay_plan_path: Path | None = None,
    replay_mixture_report_path: Path | None = None,
    replay_mixture_summary: dict[str, Any] | None = None,
    sweep_plan_path: Path | None = None,
    sweep_plan_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direct_enabled = args.direct_answer_steps > 0
    objective = {
        "target_loss": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "eval_every": args.eval_every,
            "target_loss_weight": args.target_loss_weight,
            "choice_loss_weight": args.choice_loss_weight,
            "choice_negatives": args.choice_negatives,
        },
        "direct_answer": {
            "enabled": direct_enabled,
            "steps": args.direct_answer_steps,
            "mode": args.direct_answer_mode,
            "learning_rate": args.direct_answer_learning_rate,
            "branch_position": args.direct_answer_branch_position,
            "branch_span": args.direct_answer_branch_span,
            "snapshot_mode": args.direct_answer_snapshot_mode,
            "require_branch_context_gate": (
                args.direct_answer_require_branch_context_gate
            ),
        },
        "generation": dict(generation_config),
    }
    if sweep_plan_path is not None:
        objective["controlled_sweep"] = {
            "status": "planned",
            "path": str(sweep_plan_path),
            "summary": sweep_plan_summary,
        }
    return build_training_recipe(
        version=getattr(args, "experiment_version", TRANSFORMER_RECIPE_VERSION),
        component="transformer-answer-train",
        run_id=args.run.name,
        recipe_id=transformer_training_recipe_id(args),
        purpose=(
            "Train a tiny decoder-only transformer from admitted corpus text and "
            "evaluate reliable-answer behavior under constraint-first gates."
        ),
        model={
            "architecture": TRANSFORMER_ARCHITECTURE,
            "config": dict(model_config),
            "initialization": (
                "declared QuarkLM checkpoint"
                if args.resume_checkpoint is not None
                else "random"
            ),
            "resume_checkpoint": (
                str(args.resume_checkpoint)
                if args.resume_checkpoint is not None
                else None
            ),
            "pretrained_weights": False,
        },
        tokenizer={
            "type": TRANSFORMER_TOKENIZER,
            "source": str(args.train_text),
            "vocab_size": tokenizer.vocab_size,
            "pretrained_tokenizer": False,
        },
        data={
            "train_text": str(args.train_text),
            "valid_text": str(args.valid),
            "corpus_dir": str(args.corpus_dir),
            "eval_sets": [str(path) for path in DEFAULT_ANSWER_EVALS],
            "training_examples": TRAINING_DATA_DESCRIPTION,
        },
        objective=objective,
        optimizer=dict(optimizer_config),
        replay={
            "status": "planned" if replay_plan_path is not None else "not_applicable",
            "path": str(replay_plan_path) if replay_plan_path is not None else None,
            "profile_aware": (
                direct_enabled
                and is_profile_aware_direct_answer_mode(args.direct_answer_mode)
            ),
            "mixture_report": {
                "status": (
                    "written"
                    if replay_mixture_report_path is not None
                    else "not_planned"
                ),
                "path": (
                    str(replay_mixture_report_path)
                    if replay_mixture_report_path is not None
                    else None
                ),
                "summary": replay_mixture_summary,
            },
        },
        artifacts=planned_artifacts,
        gates=acceptance_gates,
        rerun={
            "entry_point": "quark-lm-transformer answer-train",
            "arguments": {
                "train_text": str(args.train_text),
                "valid": str(args.valid),
                "corpus_dir": str(args.corpus_dir),
                "run": str(args.run),
                "steps": args.steps,
                "context_size": args.context_size,
                "embedding_dim": args.embedding_dim,
                "feedforward_dim": args.feedforward_dim,
                "direct_answer_steps": args.direct_answer_steps,
                "direct_answer_mode": args.direct_answer_mode,
                "seed": args.seed,
            },
        },
        notes=["Recipe uses admitted corpus text, corpus-trained tokenizer, and no external model."],
    )
