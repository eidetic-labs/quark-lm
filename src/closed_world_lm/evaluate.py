"""Evaluate a closed-world character model checkpoint on source probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .neural_char_model import CharMLP
from .probes import read_jsonl, score_records, summarize


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_DIR / "runs" / "latest" / "checkpoint.json"
DEFAULT_QA = PROJECT_DIR / "evals" / "qa.jsonl"
DEFAULT_UNKNOWNS = PROJECT_DIR / "evals" / "unknowns.jsonl"
DEFAULT_HELDOUT = PROJECT_DIR / "evals" / "heldout.jsonl"
DEFAULT_PARAPHRASES = PROJECT_DIR / "evals" / "paraphrases.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--unknowns", type=Path, default=DEFAULT_UNKNOWNS)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--paraphrases", type=Path, default=DEFAULT_PARAPHRASES)
    parser.add_argument("--max-new-chars", type=int, default=24)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = CharMLP.load(args.checkpoint)
    if tokenizer is None:
        raise ValueError("checkpoint does not contain a tokenizer")
    qa_records = read_jsonl(args.qa)
    unknown_records = read_jsonl(args.unknowns)
    heldout_records = read_jsonl(args.heldout)
    paraphrase_records = read_jsonl(args.paraphrases)
    candidates = sorted(
        {
            record["target"]
            for record in [*qa_records, *unknown_records, *heldout_records, *paraphrase_records]
        }
    )
    known = score_records(
        model,
        tokenizer,
        qa_records,
        args.max_new_chars,
        candidates=candidates,
    )
    unknown = score_records(
        model,
        tokenizer,
        unknown_records,
        args.max_new_chars,
        candidates=candidates,
    )
    heldout = score_records(
        model,
        tokenizer,
        heldout_records,
        args.max_new_chars,
        candidates=candidates,
    )
    paraphrases = score_records(
        model,
        tokenizer,
        paraphrase_records,
        args.max_new_chars,
        candidates=candidates,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "candidate_count": len(candidates),
        "known": summarize(known),
        "unknown": summarize(unknown),
        "heldout": summarize(heldout),
        "paraphrases": summarize(paraphrases),
        "known_records": known,
        "unknown_records": unknown,
        "heldout_records": heldout,
        "paraphrase_records": paraphrases,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    result = evaluate(parse_args(argv))
    print(json.dumps(result["known"], indent=2, sort_keys=True))
    print(json.dumps(result["unknown"], indent=2, sort_keys=True))
    print(json.dumps(result["heldout"], indent=2, sort_keys=True))
    print(json.dumps(result["paraphrases"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
