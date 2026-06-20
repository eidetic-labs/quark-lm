"""Answer-train subcommand parser for the transformer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from answer_model import DEFAULT_CORPUS_DIR, DEFAULT_TRAIN_TEXT
from curriculum import DEFAULT_OUTPUT_DIR
from transformer_cli_shared_options import (
    add_architecture_options,
    add_generation_sampling_options,
    add_optimizer_options,
    add_tokenizer_options,
)
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_experiment import TRANSFORMER_RECIPE_VERSION
from transformer_objectives import DIRECT_ANSWER_OBJECTIVE_MODES
from transformer_paths import DEFAULT_RUN_DIR
from transformer_routing_repair_bundle import EXPERIMENT_BUNDLES


def add_answer_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    answer_parser = subparsers.add_parser(
        "answer-train",
        help="train the tiny transformer on corpus-derived answer lessons",
    )
    add_answer_train_arguments(answer_parser)


def add_answer_train_arguments(answer_parser: argparse.ArgumentParser) -> None:
    answer_parser.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    answer_parser.add_argument("--valid", type=Path, default=DEFAULT_OUTPUT_DIR / "valid.txt")
    answer_parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    answer_parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    answer_parser.add_argument("--steps", type=int, default=400)
    answer_parser.add_argument("--learning-rate", type=float, default=0.04)
    answer_parser.add_argument("--target-loss-weight", type=float, default=1.0)
    answer_parser.add_argument("--choice-loss-weight", type=float, default=0.0)
    answer_parser.add_argument(
        "--choice-negatives",
        type=int,
        default=0,
        help="Wrong answer candidates sampled for each contrastive choice step.",
    )
    answer_parser.add_argument(
        "--choice-max-chars",
        type=int,
        default=0,
        help="Limit contrastive candidate loss to the first N answer chars. 0 uses the full answer.",
    )
    _add_selector_options(answer_parser)
    _add_generator_options(answer_parser)
    _add_direct_answer_options(answer_parser)
    add_architecture_options(answer_parser)
    add_tokenizer_options(answer_parser)
    answer_parser.add_argument("--seed", type=int, default=17)
    answer_parser.add_argument("--eval-every", type=int, default=100)
    answer_parser.add_argument("--max-new-chars", type=int, default=48)
    add_generation_sampling_options(answer_parser)
    add_optimizer_options(answer_parser)
    answer_parser.add_argument("--resume-checkpoint", type=Path, default=None)
    answer_parser.add_argument("--resume-optimizer", type=Path, default=None)
    answer_parser.add_argument(
        "--candidate-scope",
        choices=["all", "eval"],
        default="eval",
        help="Candidate set for answer snapshots. 'eval' scores against targets in the current eval set.",
    )
    answer_parser.add_argument(
        "--include-completions",
        action="store_true",
        help="Generate free-form completions during answer snapshots. Slower, but records exact generation.",
    )
    _add_experiment_options(answer_parser)


def _add_selector_options(answer_parser: argparse.ArgumentParser) -> None:
    answer_parser.add_argument(
        "--selector-steps",
        type=int,
        default=0,
        help="Train a closed-world answer candidate selector alongside transformer evidence.",
    )
    answer_parser.add_argument("--selector-learning-rate", type=float, default=0.08)
    answer_parser.add_argument(
        "--selector-negatives",
        type=int,
        default=0,
        help="Wrong selector candidates sampled per selector step. 0 trains against all labels.",
    )
    answer_parser.add_argument("--selector-eval-every", type=int, default=200)
    answer_parser.add_argument(
        "--selector-emit-completions",
        action="store_true",
        help="Record selector-chosen candidates as emitted completions for exact-match evidence.",
    )


def _add_generator_options(answer_parser: argparse.ArgumentParser) -> None:
    answer_parser.add_argument(
        "--generator-steps",
        type=int,
        default=0,
        help="Train a transformer-guided character answer generator without answer candidates.",
    )
    answer_parser.add_argument("--generator-learning-rate", type=float, default=0.08)
    answer_parser.add_argument("--generator-eval-every", type=int, default=200)
    answer_parser.add_argument("--generator-max-answer-chars", type=int, default=64)
    answer_parser.add_argument("--generator-transformer-top-k", type=int, default=3)


def _add_direct_answer_options(answer_parser: argparse.ArgumentParser) -> None:
    answer_parser.add_argument(
        "--direct-answer-steps",
        type=int,
        default=0,
        help="Continue training transformer weights for greedy answer completion.",
    )
    answer_parser.add_argument("--direct-answer-learning-rate", type=float, default=0.035)
    answer_parser.add_argument("--direct-answer-eval-every", type=int, default=200)
    answer_parser.add_argument("--direct-answer-max-new-chars", type=int, default=96)
    answer_parser.add_argument(
        "--direct-answer-snapshot-mode",
        choices=["full", "branch-only"],
        default="full",
        help=(
            "Direct-answer JSONL snapshot detail. 'branch-only' skips greedy "
            "completion evals and records only branch profiles and branch-context "
            "coverage for bounded screening runs."
        ),
    )
    answer_parser.add_argument(
        "--direct-answer-mode",
        choices=DIRECT_ANSWER_OBJECTIVE_MODES,
        default="first-error",
        help="Direct transformer update policy for greedy answer completion.",
    )
    answer_parser.add_argument(
        "--memory-consolidation-source-plan",
        type=Path,
        default=None,
        help=(
            "Prior memory_consolidation_plan.json to consume for gated "
            "memory-backed direct-answer consolidation modes."
        ),
    )
    answer_parser.add_argument(
        "--direct-answer-frontier-metrics",
        type=Path,
        default=None,
        help=(
            "Prior transformer_answer_metrics.json used as an explicit "
            "frontier reference for evidence gates. The file is not training data."
        ),
    )
    answer_parser.add_argument(
        "--memory-consolidation-max-profiles",
        type=int,
        default=3,
        help="Maximum memory-backed failed profiles to consume from the source plan.",
    )
    answer_parser.add_argument("--direct-answer-negative-weight", type=float, default=0.5)
    answer_parser.add_argument("--direct-answer-positive-weight", type=float, default=1.0)
    answer_parser.add_argument("--direct-answer-contrast-weight", type=float, default=1.0)
    answer_parser.add_argument("--direct-answer-recovery-steps", type=int, default=3)
    answer_parser.add_argument("--direct-answer-branch-position", type=int, default=1)
    answer_parser.add_argument("--direct-answer-branch-span", type=int, default=1)
    answer_parser.add_argument("--direct-answer-branch-batch-size", type=int, default=4)
    answer_parser.add_argument(
        "--direct-answer-repair-target-profile",
        action="append",
        default=None,
        help=(
            "Limit declared profile-balanced repair batches to this trainable "
            "profile. Repeat for multiple profiles."
        ),
    )
    answer_parser.add_argument("--direct-answer-hard-negatives", type=int, default=16)
    answer_parser.add_argument("--direct-answer-train-top-layer-only", action="store_true")
    answer_parser.add_argument(
        "--direct-answer-freeze-output-bias",
        action="store_true",
        help=(
            "Exclude the transformer output bias from direct-answer updates so "
            "branch screens cannot improve loss by moving one global token bias."
        ),
    )
    answer_parser.add_argument(
        "--direct-answer-restore-best-branch-snapshot",
        action="store_true",
        help=(
            "Restore the direct-answer weights with the best branch-diversity "
            "snapshot score before final metrics and checkpoint writing."
        ),
    )
    answer_parser.add_argument(
        "--direct-answer-require-branch-context-gate",
        action="store_true",
        help=(
            "Skip direct-answer training unless branch contexts have complete "
            "semantic coverage, no ambiguous target-token contexts, and no skipped records."
        ),
    )
    answer_parser.add_argument(
        "--skip-post-direct-snapshot",
        action="store_true",
        help=(
            "Skip the full answer-candidate snapshot after direct-answer updates. "
            "Use only for bounded screening runs; promotion evidence should keep "
            "the default full post-direct snapshot."
        ),
    )
    answer_parser.add_argument("--direct-answer-sequence-interval", type=int, default=50)
    answer_parser.add_argument("--direct-answer-rollout-interval", type=int, default=5)
    answer_parser.add_argument(
        "--direct-answer-terminator",
        type=str,
        default=ANSWER_TERMINATOR,
        help="Single admitted character that stops direct answer generation.",
    )


def _add_experiment_options(answer_parser: argparse.ArgumentParser) -> None:
    answer_parser.add_argument("--experiment-version", default=TRANSFORMER_RECIPE_VERSION)
    answer_parser.add_argument(
        "--experiment-bundle",
        choices=EXPERIMENT_BUNDLES,
        default=None,
        help="Declared experiment bundle that adds required acceptance gates.",
    )
    answer_parser.add_argument("--experiment-hypothesis", default=None)
    answer_parser.add_argument(
        "--experiment-acceptance-gate",
        action="append",
        default=None,
        help="Additional required experiment gate formatted as name:rule.",
    )
    answer_parser.add_argument(
        "--experiment-failure-criterion",
        action="append",
        default=None,
        help="Additional failure criterion for this screen.",
    )
    answer_parser.add_argument("--experiment-note", action="append", default=None)
