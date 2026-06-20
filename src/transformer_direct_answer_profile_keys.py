"""Training-profile keys for direct-answer branch routing."""

from __future__ import annotations

from typing import Any


SOURCE_SUFFIX_PROFILE_KEYS = {"glossary", "learning", "owner", "self"}


def direct_answer_training_profile_key(example: Any) -> str:
    """Return the trainable profile family for a direct-answer example."""

    source = str(getattr(example, "source", "") or "")
    if not source:
        return "unknown"
    source_kind, suffix = _source_parts(source)
    if suffix in SOURCE_SUFFIX_PROFILE_KEYS:
        return suffix
    if source_kind == "unknown":
        return "unknowns"
    if source_kind in {"bridge", "fact", "qa"}:
        return "qa"
    return source_kind


def trainable_eval_profile_keys() -> set[str]:
    """Return eval profile names that have admitted training-family analogs."""

    return {"glossary", "learning", "owner", "qa", "self", "unknowns"}


def _source_parts(source: str) -> tuple[str, str]:
    parts = source.split(":", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]
