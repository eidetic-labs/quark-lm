"""Train the dependency-free closed-world character model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .curriculum import DEFAULT_OUTPUT_DIR, build_curriculum, write_curriculum
from .neural_char_model import CharMLP, ModelConfig, average_nll, context_before
from .probes import read_jsonl, target_nll_summary
from .tokenizer import CharTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "latest"
DEFAULT_PROBES = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_OUTPUT_DIR / "train.txt")
    parser.add_argument("--valid", type=Path, default=DEFAULT_OUTPUT_DIR / "valid.txt")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--valid-limit", type=int, default=2000)
    parser.add_argument(
        "--probe",
        action="append",
        type=Path,
        default=None,
        help="JSONL probe file to score during training. Defaults to all evals/*.jsonl.",
    )
    return parser.parse_args(argv)


def ensure_curriculum(corpus_path: Path, valid_path: Path) -> None:
    if corpus_path.exists() and valid_path.exists():
        return
    curriculum = build_curriculum()
    write_curriculum(curriculum, DEFAULT_OUTPUT_DIR)


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    ensure_curriculum(args.corpus, args.valid)
    train_text = args.corpus.read_text(encoding="utf-8")
    valid_text = args.valid.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.train(train_text)
    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    model = CharMLP.init_random(config)
    rng = random.Random(args.seed)
    probe_paths = args.probe if args.probe is not None else DEFAULT_PROBES
    probes = {path.stem: read_jsonl(path) for path in probe_paths if path.exists()}

    running_loss = 0.0
    metrics: dict[str, float | int | str] = {
        "steps": args.steps,
        "train_chars": len(train_text),
        "valid_chars": len(valid_text),
        "vocab_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
    }
    history_path = args.run / "metrics.jsonl"
    args.run.mkdir(parents=True, exist_ok=True)

    def write_history(step: int, train_nll: float | None) -> dict[str, object]:
        record: dict[str, object] = {
            "step": step,
            "train_nll": train_nll,
            "valid_nll": average_nll(model, valid_ids, tokenizer.pad_id, args.valid_limit),
            "probes": {
                name: target_nll_summary(model, tokenizer, records)
                for name, records in sorted(probes.items())
            },
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    baseline_record = write_history(step=0, train_nll=None)
    last_history_step = 0
    last_history: dict[str, object] = baseline_record

    for step in range(1, args.steps + 1):
        position = rng.randrange(len(train_ids))
        context = context_before(train_ids, position, args.context_size, tokenizer.pad_id)
        loss = model.train_step(context, train_ids[position], args.learning_rate)
        running_loss += loss
        if args.eval_every > 0 and step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            history = write_history(step=step, train_nll=train_loss)
            last_history = history
            last_history_step = step
            valid_loss = history["valid_nll"]
            print(f"step={step} train_nll={train_loss:.4f} valid_nll={valid_loss:.4f}")
            running_loss = 0.0

    checkpoint_path = args.run / "checkpoint.json"
    model.save(checkpoint_path, tokenizer)
    tokenizer.save(args.run / "tokenizer.json")

    if last_history_step == args.steps:
        final_history = last_history
    else:
        final_history = write_history(step=args.steps, train_nll=None)
    final_valid_loss = float(final_history["valid_nll"])
    metrics["baseline_valid_nll"] = baseline_record["valid_nll"]  # type: ignore[assignment]
    metrics["final_valid_nll"] = final_valid_loss
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["history"] = str(history_path)
    with (args.run / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
