"""Dependency-free tiny decoder-only transformer language model.

The implementation is intentionally small and auditable. It uses learned token
and position embeddings, one causal self-attention block, a feed-forward block,
and a next-character language-model head. All weights start from random values;
the tokenizer is trained from admitted corpus text.
"""

from __future__ import annotations

import argparse
from typing import Any

from tokenizer import CharTokenizer
from transformer_answer_training import train_transformer_answers_command
from transformer_cli import run_transformer_cli
from transformer_incremental_update_command import incremental_update_command
import transformer_model as transformer_model_types
from transformer_text_commands import (
    eval_transformer_command,
    initialize_transformer_for_training_command,
    train_transformer_command,
)
from transformer_tiny_lm import TinyTransformerLM

GenerationConfig = transformer_model_types.GenerationConfig
OptimizationConfig = transformer_model_types.OptimizationConfig
TransformerConfig = transformer_model_types.TransformerConfig

__all__ = [
    "GenerationConfig",
    "OptimizationConfig",
    "TinyTransformerLM",
    "TransformerConfig",
    "eval_transformer",
    "initialize_transformer_for_training",
    "incremental_update",
    "main",
    "train_transformer",
    "train_transformer_answers",
]


def initialize_transformer_for_training(
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
) -> tuple[TinyTransformerLM, dict[str, Any]]:
    return initialize_transformer_for_training_command(args, tokenizer, TinyTransformerLM)


def train_transformer(args: argparse.Namespace) -> dict[str, Any]:
    return train_transformer_command(args, TinyTransformerLM)


def eval_transformer(args: argparse.Namespace) -> dict[str, Any]:
    return eval_transformer_command(args, TinyTransformerLM)


def train_transformer_answers(args: argparse.Namespace) -> dict[str, Any]:
    return train_transformer_answers_command(
        args,
        TinyTransformerLM,
        initialize_transformer_for_training,
    )


def incremental_update(args: argparse.Namespace) -> dict[str, Any]:
    return incremental_update_command(args, TinyTransformerLM)


def main(argv: list[str] | None = None) -> int:
    return run_transformer_cli(
        argv,
        train_transformer=train_transformer,
        eval_transformer=eval_transformer,
        train_transformer_answers=train_transformer_answers,
        incremental_update=incremental_update,
    )


if __name__ == "__main__":
    raise SystemExit(main())
