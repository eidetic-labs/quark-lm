"""Evaluation-set duplicate and train/eval leakage summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from corpus_artifacts import read_jsonl
from corpus_duplicate_summary import duplicate_values
from corpus_example_summary import example_value


def train_eval_overlap(
    train_text: str,
    examples: list[Any],
    eval_paths: list[Path],
) -> dict[str, Any]:
    prompts_to_sources: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        prompts_to_sources[example_value(example, "prompt")].add(
            example_value(example, "source", "unknown")
        )

    eval_summaries: dict[str, Any] = {}
    protected_total = 0
    protected_overlaps = 0
    protected_train_text_overlaps = 0
    for path in eval_paths:
        records = read_jsonl(path)
        prompt_overlaps: list[dict[str, Any]] = []
        train_text_overlaps: list[dict[str, Any]] = []
        protected_prompt_overlaps: list[dict[str, Any]] = []
        protected_train_text_prompt_overlaps: list[dict[str, Any]] = []
        for record in records:
            prompt = str(record.get("prompt", ""))
            record_id = str(record.get("id", ""))
            protected = path.stem == "heldout" or (
                path.stem == "owner" and "-heldout-" in record_id
            )
            if protected:
                protected_total += 1
            if prompt in prompts_to_sources:
                overlap = {
                    "id": record_id,
                    "prompt": prompt,
                    "sources": sorted(prompts_to_sources[prompt]),
                    "protected": protected,
                }
                prompt_overlaps.append(overlap)
                if protected:
                    protected_prompt_overlaps.append(overlap)
                    protected_overlaps += 1
            if prompt and prompt in train_text:
                overlap = {
                    "id": record_id,
                    "prompt": prompt,
                    "protected": protected,
                }
                train_text_overlaps.append(overlap)
                if protected:
                    protected_train_text_prompt_overlaps.append(overlap)
                    protected_train_text_overlaps += 1
        eval_summaries[path.stem] = {
            "path": str(path),
            "record_count": len(records),
            "prompt_overlap_count": len(prompt_overlaps),
            "train_text_prompt_overlap_count": len(train_text_overlaps),
            "protected_prompt_overlap_count": len(protected_prompt_overlaps),
            "protected_train_text_prompt_overlap_count": len(
                protected_train_text_prompt_overlaps
            ),
            "prompt_overlaps": prompt_overlaps[:20],
            "train_text_prompt_overlaps": train_text_overlaps[:20],
            "protected_prompt_overlaps": protected_prompt_overlaps[:20],
            "protected_train_text_prompt_overlaps": (
                protected_train_text_prompt_overlaps[:20]
            ),
        }
    return {
        "eval_sets": eval_summaries,
        "protected_eval_records": protected_total,
        "protected_prompt_overlap_count": protected_overlaps,
        "protected_train_text_prompt_overlap_count": protected_train_text_overlaps,
        "passed": protected_overlaps == 0 and protected_train_text_overlaps == 0,
    }


def eval_duplicate_summary(eval_paths: list[Path]) -> dict[str, Any]:
    per_set = {
        path.stem: duplicate_values(read_jsonl(path), "id")
        for path in eval_paths
    }
    all_records: list[dict[str, Any]] = []
    for path in eval_paths:
        for record in read_jsonl(path):
            all_records.append({**record, "eval_set": path.stem})
    return {
        "per_eval_set": per_set,
        "global_eval_ids": duplicate_values(all_records, "id"),
    }


def eval_set_counts(eval_paths: list[Path]) -> dict[str, Any]:
    return {
        path.stem: {
            "path": str(path),
            "records": len(read_jsonl(path)),
        }
        for path in eval_paths
    }
