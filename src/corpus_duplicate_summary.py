"""Duplicate-detection summaries for corpus and evaluation records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from corpus_example_summary import example_value


def duplicate_values(records: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    positions: dict[str, list[int]] = defaultdict(list)
    missing = 0
    for index, record in enumerate(records):
        value = record.get(field_name)
        if value is None:
            missing += 1
            continue
        positions[str(value)].append(index)
    duplicates = [
        {"value": value, "count": len(indexes), "indexes": indexes}
        for value, indexes in sorted(positions.items())
        if len(indexes) > 1
    ]
    return {
        "field": field_name,
        "record_count": len(records),
        "missing_count": missing,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "passed": not duplicates and missing == 0,
    }


def duplicate_example_pairs(examples: list[Any]) -> dict[str, Any]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        prompt = example_value(example, "prompt")
        target = example_value(example, "target")
        positions[f"{prompt}\n=>{target}"].append(index)
    duplicates = [
        {"key": key, "count": len(indexes), "indexes": indexes}
        for key, indexes in sorted(positions.items())
        if len(indexes) > 1
    ]
    return {
        "field": "prompt+target",
        "record_count": len(examples),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "passed": not duplicates,
    }
