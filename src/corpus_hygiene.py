"""Compatibility facade for corpus hygiene and training-plan artifacts."""

from __future__ import annotations

from corpus_artifacts import SCHEMA_VERSION, read_json, read_jsonl, write_json_artifact
from corpus_duplicate_summary import duplicate_example_pairs, duplicate_values
from corpus_eval_summary import (
    eval_duplicate_summary,
    eval_set_counts,
    train_eval_overlap,
)
from corpus_example_summary import (
    example_value,
    rare_profile_coverage,
    source_family,
    source_mixture,
    source_target,
)
from corpus_hygiene_report import build_corpus_hygiene_report
from corpus_source_summary import corpus_source_summary
from corpus_training_plan import attach_replay_plan_summary, build_training_plan


__all__ = [
    "SCHEMA_VERSION",
    "attach_replay_plan_summary",
    "build_corpus_hygiene_report",
    "build_training_plan",
    "corpus_source_summary",
    "duplicate_example_pairs",
    "duplicate_values",
    "eval_duplicate_summary",
    "eval_set_counts",
    "example_value",
    "rare_profile_coverage",
    "read_json",
    "read_jsonl",
    "source_family",
    "source_mixture",
    "source_target",
    "train_eval_overlap",
    "write_json_artifact",
]
