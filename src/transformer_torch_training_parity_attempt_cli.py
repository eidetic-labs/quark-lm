"""CLI for optional PyTorch training parity attempt artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curriculum import DEFAULT_CORPUS_DIR
from transformer_torch_training_parity_attempt import (
    DEFAULT_OUTPUT_DIR,
    build_torch_training_parity_attempt,
    write_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_audit import (
    build_torch_training_parity_attempt_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="validate an existing output directory without rebuilding artifacts",
    )
    parser.add_argument(
        "--fixture-id",
        default="admitted-curriculum-training-parity",
    )
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--context-index", type=int, default=4)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--requested-device", default="cpu")
    parser.add_argument("--requested-dtype", default="float64")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_existing:
        audit = build_torch_training_parity_attempt_audit(args.output_dir)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["passed"] else 1
    artifacts = build_torch_training_parity_attempt(
        corpus_dir=args.corpus_dir,
        fixture_id=args.fixture_id,
        seed=args.seed,
        context_index=args.context_index,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        feedforward_dim=args.feedforward_dim,
        learning_rate=args.learning_rate,
        steps=args.steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        requested_device=args.requested_device,
        requested_dtype=args.requested_dtype,
    )
    attempt = write_torch_training_parity_attempt(args.output_dir, artifacts)
    print(json.dumps(attempt, indent=2, sort_keys=True))
    return 0 if attempt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
