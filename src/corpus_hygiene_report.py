"""Assembly of corpus hygiene reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_artifacts import SCHEMA_VERSION, read_jsonl
from corpus_duplicate_summary import duplicate_example_pairs, duplicate_values
from corpus_eval_summary import eval_duplicate_summary, train_eval_overlap
from corpus_example_summary import rare_profile_coverage, source_mixture
from corpus_source_summary import corpus_source_summary


def build_corpus_hygiene_report(
    component: str,
    corpus_dir: Path,
    train_text_path: Path,
    eval_paths: list[Path],
    training_examples: list[Any],
    rare_profile_min_count: int = 3,
) -> dict[str, Any]:
    train_text = train_text_path.read_text(encoding="utf-8") if train_text_path.exists() else ""
    admissions = read_jsonl(corpus_dir / "admissions.jsonl")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "corpus_hygiene_report",
        "component": component,
        "corpus_sources": corpus_source_summary(corpus_dir),
        "training_text": {
            "path": str(train_text_path),
            "chars": len(train_text),
        },
        "training_examples": {
            "count": len(training_examples),
            "source_mixture": source_mixture(training_examples),
            "duplicates": duplicate_example_pairs(training_examples),
            "rare_profile_coverage": rare_profile_coverage(
                training_examples,
                rare_profile_min_count,
            ),
        },
        "duplicate_ids": {
            "admissions": duplicate_values(admissions, "id"),
            "evals": eval_duplicate_summary(eval_paths),
        },
        "train_eval_overlap": train_eval_overlap(
            train_text,
            training_examples,
            eval_paths,
        ),
        "candidate_ratio": source_mixture(training_examples)["candidate_ratio"],
    }
