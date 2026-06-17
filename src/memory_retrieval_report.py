"""Retrieval-memory report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from curriculum import DEFAULT_CORPUS_DIR, PROJECT_DIR, read_jsonl
from memory_cards import MemoryCard
from memory_index import ClosedWorldMemoryIndex


SCHEMA_VERSION = 1
REPORT_KIND = "retrieval_memory_report"
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


def memory_summary(cards: list[MemoryCard]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    profiles: dict[str, int] = {}
    for card in cards:
        sources[card.source] = sources.get(card.source, 0) + 1
        profiles[card.profile] = profiles.get(card.profile, 0) + 1
    return {
        "card_count": len(cards),
        "source_counts": dict(sorted(sources.items())),
        "profile_counts": dict(sorted(profiles.items())),
    }


def build_retrieval_memory_report(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    eval_paths: list[Path] | None = None,
) -> dict[str, Any]:
    index = ClosedWorldMemoryIndex.from_corpus(corpus_dir)
    evals = {
        path.stem: index.evaluate_records(read_jsonl(path))
        for path in (eval_paths or DEFAULT_EVALS)
        if path.exists()
    }
    total_count = sum(summary["count"] for summary in evals.values())
    total_exact = sum(summary["exact"] for summary in evals.values())
    failed_by_eval = {
        name: [record["id"] for record in summary["failed_records"]]
        for name, summary in evals.items()
        if summary["failed_records"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "corpus_dir": str(corpus_dir),
        "dataset_exclusivity": {
            "uses_external_model": False,
            "external_embeddings": False,
            "pretrained_retriever": False,
            "updates_weights": False,
            "memory_source": "ledgered closed-world corpus",
        },
        "memory": memory_summary(index.cards),
        "evals": evals,
        "summary": {
            "eval_count": len(evals),
            "record_count": total_count,
            "exact": total_exact,
            "exact_rate": total_exact / total_count if total_count else 0.0,
            "failed_by_eval": failed_by_eval,
        },
        "self_improvement": {
            "status": "memory_serves_before_weight_consolidation",
            "rule": (
                "New knowledge can be retrieved immediately after corpus admission; "
                "weight updates remain a gated consolidation step."
            ),
            "next_weight_step": (
                "Use failed retrieval records and neural branch-diversity failures "
                "to decide which memories need consolidation into weights."
            ),
        },
    }


def write_retrieval_memory_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
