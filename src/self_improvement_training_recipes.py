"""Training recipe assembly for self-improvement cycles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from self_improvement_experiment_contract import (
    SELF_IMPROVEMENT_COMPONENT,
    SELF_IMPROVEMENT_RECIPE_ID,
    allowed_data_sources,
    experiment_run_id,
    experiment_version,
)
from training_recipe_core import build_training_recipe


def self_improvement_training_recipe(
    args: argparse.Namespace,
    run_dir: Path,
    attempt_dir: Path,
    train_text_path: Path,
    planned_artifacts: list[Path],
    acceptance_gates: list[dict[str, Any]],
    tokenizer_candidate_summary: dict[str, Any] | None = None,
    tokenizer_manifest_path: Path | None = None,
    tokenizer_report_path: Path | None = None,
) -> dict[str, Any]:
    tokenizer = {
        "type": "tokenizer.CharTokenizer",
        "source": str(train_text_path),
        "pretrained_tokenizer": False,
        "active_tokenizer_changed": False,
    }
    if tokenizer_candidate_summary is not None:
        tokenizer["candidate"] = {
            "type": tokenizer_candidate_summary.get("tokenizer_type"),
            "manifest_path": (
                str(tokenizer_manifest_path)
                if tokenizer_manifest_path is not None
                else None
            ),
            "report_path": (
                str(tokenizer_report_path)
                if tokenizer_report_path is not None
                else None
            ),
            "summary": tokenizer_candidate_summary,
            "promotion_status": "candidate_only_not_promoted",
        }
    return build_training_recipe(
        version=experiment_version(args),
        component=SELF_IMPROVEMENT_COMPONENT,
        run_id=experiment_run_id(run_dir, attempt_dir),
        recipe_id=SELF_IMPROVEMENT_RECIPE_ID,
        purpose=(
            "Train answer_model and answer_decoder from admitted closed-world "
            "curriculum while preserving verifier, leakage, forgetting, and exact eval gates."
        ),
        model={
            "components": ["answer_model", "answer_decoder"],
            "initialization": "random or declared QuarkLM checkpoint only",
            "pretrained_weights": False,
        },
        tokenizer=tokenizer,
        data={
            "corpus_dir": str(args.corpus_dir),
            "train_text": str(train_text_path),
            "allowed_sources": allowed_data_sources(args, train_text_path),
        },
        objective={
            "answer_model": {
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "eval_every": args.eval_every,
            },
            "answer_decoder": {
                "steps": args.decoder_steps,
                "learning_rate": args.decoder_learning_rate,
                "eval_every": args.decoder_eval_every,
                "max_answer_chars": args.max_answer_chars,
            },
        },
        optimizer={
            "answer_model": "answer_model train_model",
            "answer_decoder": "answer_decoder train_decoder",
            "seed": args.seed,
        },
        artifacts=planned_artifacts,
        gates=acceptance_gates,
        replay={"status": "not_applicable"},
        rerun={
            "entry_point": "quark-lm-self-improve answer-cycle",
            "arguments": {
                "corpus_dir": str(args.corpus_dir),
                "build_dir": str(args.build_dir),
                "run": str(run_dir),
                "steps": args.steps,
                "decoder_steps": args.decoder_steps,
                "seed": args.seed,
            },
        },
        notes=[
            "Recipe uses admitted curriculum only and no external model.",
            "Tokenizer candidates are recorded as evidence before any promotion.",
        ],
    )
