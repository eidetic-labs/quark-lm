"""Replay-mixture evidence for transformer answer-training screens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from corpus_example_summary import source_mixture
from transformer_replay_mixture_buckets import (
    replay_mixture_bucket_summary,
    replay_mixture_buckets,
)


REPLAY_MIXTURE_KIND = "transformer_replay_mixture_report"
REPLAY_MIXTURE_SCHEMA_VERSION = 1


def build_transformer_replay_mixture_report(
    *,
    run_id: str,
    train_text_path: Path,
    examples: list[Any],
    training_pool: list[Any],
    eval_records: dict[str, list[dict[str, Any]]],
    admissions: list[dict[str, Any]],
) -> dict[str, Any]:
    buckets = replay_mixture_buckets(examples, eval_records, admissions)
    return {
        "schema_version": REPLAY_MIXTURE_SCHEMA_VERSION,
        "kind": REPLAY_MIXTURE_KIND,
        "component": "transformer-answer-train",
        "run_id": run_id,
        "train_text": str(train_text_path),
        "rule": (
            "Every transformer screen must declare which evidence mixture it "
            "trained or evaluated against so current-task gains cannot hide "
            "retention, unknown-policy, tokenizer-stress, or heldout-paraphrase gaps."
        ),
        "training_examples": source_mixture(examples),
        "scheduled_training_pool": source_mixture(training_pool),
        "eval_sets": {
            name: {
                "count": len(records),
                "target_count": len({record.get("target", "") for record in records}),
            }
            for name, records in sorted(eval_records.items())
        },
        "buckets": buckets,
        "summary": replay_mixture_bucket_summary(buckets),
    }


def replay_mixture_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary", {}))
    return {
        "kind": report.get("kind"),
        "bucket_count": summary.get("bucket_count", 0),
        "non_empty_bucket_count": summary.get("non_empty_bucket_count", 0),
        "missing_required_buckets": summary.get("missing_required_buckets", []),
        "passed": summary.get("passed") is True,
    }


def write_transformer_replay_mixture_report(
    path: Path,
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
