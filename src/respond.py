"""Reliable closed-world responses learned from admitted corpus facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus_responder import CorpusResponder, DEFAULT_TRAIN_TEXT, FactRecord
from probes import read_jsonl


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVALS = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "paraphrases.jsonl",
    PROJECT_DIR / "evals" / "owner.jsonl",
    PROJECT_DIR / "evals" / "self.jsonl",
    PROJECT_DIR / "evals" / "learning.jsonl",
    PROJECT_DIR / "evals" / "admissions.jsonl",
    PROJECT_DIR / "evals" / "admission_paraphrases.jsonl",
    PROJECT_DIR / "evals" / "glossary.jsonl",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    responder = CorpusResponder.load_train_text(args.train_text)
    if args.question:
        prompt = f"question: {args.question}\nanswer:"
        print(responder.answer_prompt(prompt))
        return 0

    if args.eval:
        result = {
            path.stem: responder.evaluate(read_jsonl(path))
            for path in DEFAULT_EVALS
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")
        summary = {
            name: {
                "count": value["count"],
                "exact": value["exact"],
                "exact_rate": value["exact_rate"],
            }
            for name, value in result.items()
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    raise SystemExit("pass --question or --eval")


if __name__ == "__main__":
    raise SystemExit(main())
