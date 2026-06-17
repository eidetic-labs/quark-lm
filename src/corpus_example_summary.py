"""Summaries for closed-world training examples and their source labels."""

from __future__ import annotations

from collections import Counter
from typing import Any


def example_value(example: Any, field_name: str, default: str = "") -> str:
    if isinstance(example, dict):
        value = example.get(field_name, default)
    else:
        value = getattr(example, field_name, default)
    return str(value) if value is not None else default


def source_family(source: str) -> str:
    return source.split(":", 1)[0] if source else "unknown"


def source_target(source: str) -> str:
    return source.split(":", 1)[1] if ":" in source else "unknown"


def source_mixture(examples: list[Any]) -> dict[str, Any]:
    sources = Counter(example_value(example, "source", "unknown") for example in examples)
    families = Counter(source_family(source) for source in sources.elements())
    targets = Counter(source_target(source) for source in sources.elements())
    candidate_count = sum(
        count
        for source, count in sources.items()
        if source_family(source) == "candidate"
    )
    total = len(examples)
    return {
        "total_examples": total,
        "by_source": dict(sorted(sources.items())),
        "by_family": dict(sorted(families.items())),
        "by_target": dict(sorted(targets.items())),
        "candidate_examples": candidate_count,
        "candidate_ratio": candidate_count / total if total else 0.0,
    }


def rare_profile_coverage(
    examples: list[Any],
    min_count: int = 3,
) -> dict[str, Any]:
    counts = Counter(example_value(example, "source", "unknown") for example in examples)
    rare = [
        {"profile": profile, "count": count}
        for profile, count in sorted(counts.items())
        if count < min_count
    ]
    return {
        "profile_count": len(counts),
        "min_count": min_count,
        "rare_profile_count": len(rare),
        "rare_profiles": rare,
        "passed": not rare,
    }
