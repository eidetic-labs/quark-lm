"""Deterministic closed-world memory retrieval from the admitted corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curriculum import DEFAULT_CORPUS_DIR
from memory_cards import (
    MemoryCard,
    build_memory_cards,
    fact_memory_cards,
    glossary_memory_cards,
    learning_memory_cards,
    self_memory_cards,
)
from memory_index import ClosedWorldMemoryIndex, UNKNOWN_ANSWER
from memory_retrieval_report import (
    DEFAULT_EVALS,
    REPORT_KIND,
    SCHEMA_VERSION,
    build_retrieval_memory_report,
    memory_summary,
    write_retrieval_memory_report,
)
from memory_retrieval_signatures import (
    prompt_signature,
    signatures_match,
    tokenize,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_retrieval_memory_report(args.corpus_dir)
    if args.output is not None:
        write_retrieval_memory_report(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["failed_by_eval"] == {} else 1


if __name__ == "__main__":
    raise SystemExit(main())
