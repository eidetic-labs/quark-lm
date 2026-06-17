"""Deterministic closed-world memory retrieval from the admitted corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curriculum import DEFAULT_CORPUS_DIR
from memory_index import ClosedWorldMemoryIndex
from memory_retrieval_report import (
    build_retrieval_memory_report,
    write_retrieval_memory_report,
)


__all__ = [
    "ClosedWorldMemoryIndex",
    "build_retrieval_memory_report",
    "write_retrieval_memory_report",
]


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
