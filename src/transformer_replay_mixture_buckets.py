"""Bucket construction for transformer replay-mixture reports."""

from __future__ import annotations

from collections import Counter
from typing import Any

from corpus_example_summary import source_family, source_target


def replay_mixture_buckets(
    examples: list[Any],
    eval_records: dict[str, list[dict[str, Any]]],
    admissions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    admission_keys = _admission_keys(admissions)
    return {
        "new_lessons": _new_lesson_bucket(examples, admission_keys),
        "prior_accepted_facts": _prior_fact_bucket(examples, admission_keys),
        "glossary_self_facts": _source_target_bucket(
            examples,
            {"glossary", "self", "learning"},
        ),
        "unknown_policy_probes": _unknown_policy_bucket(examples, eval_records),
        "heldout_paraphrases": _heldout_paraphrase_bucket(eval_records),
        "tokenizer_stress_strings": _tokenizer_stress_bucket(
            examples,
            eval_records,
        ),
    }


def replay_mixture_bucket_summary(
    buckets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = sorted(buckets)
    missing = [
        name
        for name in required
        if buckets[name].get("total_count", buckets[name].get("count", 0)) == 0
    ]
    return {
        "bucket_count": len(buckets),
        "non_empty_bucket_count": len(required) - len(missing),
        "missing_required_buckets": missing,
        "passed": not missing,
    }


def _new_lesson_bucket(
    examples: list[Any],
    admission_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    matched = [
        example
        for example in examples
        if _matches_any_admission(example, admission_keys)
    ]
    return _example_bucket(
        matched,
        "Admitted-memory lessons converted into training examples.",
    )


def _prior_fact_bucket(
    examples: list[Any],
    admission_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    matched = [
        example
        for example in examples
        if _is_prior_fact_example(example)
        and not _matches_any_admission(example, admission_keys)
    ]
    return _example_bucket(
        matched,
        "Original ledgered story facts and bridge examples used for retention.",
    )


def _source_target_bucket(
    examples: list[Any],
    targets: set[str],
) -> dict[str, Any]:
    matched = [
        example
        for example in examples
        if source_target(_example_source(example)) in targets
    ]
    return _example_bucket(
        matched,
        "Glossary, self-knowledge, and learning-process facts.",
    )


def _unknown_policy_bucket(
    examples: list[Any],
    eval_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    training = [
        example
        for example in examples
        if source_family(_example_source(example)) == "unknown"
        or _example_target(example) == " unknown."
    ]
    eval_count = sum(
        len(records)
        for name, records in eval_records.items()
        if "unknown" in name
    )
    bucket = _example_bucket(
        training,
        "Unknown-policy training examples and held-out unknown probes.",
    )
    bucket["eval_record_count"] = eval_count
    bucket["total_count"] = bucket["count"] + eval_count
    return bucket


def _heldout_paraphrase_bucket(
    eval_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    names = [
        name
        for name in sorted(eval_records)
        if "heldout" in name or "paraphrase" in name
    ]
    return {
        "description": "Evaluation-only heldout and paraphrase probes.",
        "count": sum(len(eval_records[name]) for name in names),
        "eval_sets": names,
        "training_data": False,
    }


def _tokenizer_stress_bucket(
    examples: list[Any],
    eval_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    strings = {
        _example_target(example).strip()
        for example in examples
        if len(_example_target(example).strip()) >= 8
    }
    for records in eval_records.values():
        for record in records:
            target = str(record.get("target", "")).strip()
            if len(target) >= 8:
                strings.add(target)
    samples = sorted(strings, key=lambda item: (-len(item), item))[:16]
    return {
        "description": "Longer corpus-derived answer strings used to stress tokenizer compression.",
        "count": len(strings),
        "sample_count": len(samples),
        "samples": samples,
        "training_data": False,
    }


def _example_bucket(examples: list[Any], description: str) -> dict[str, Any]:
    sources = Counter(_example_source(example) for example in examples)
    targets = Counter(source_target(source) for source in sources.elements())
    return {
        "description": description,
        "count": len(examples),
        "by_source": dict(sorted(sources.items())),
        "by_target": dict(sorted(targets.items())),
        "training_data": True,
    }


def _admission_keys(admissions: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(record.get("person", "")), str(record.get("object", "")))
        for record in admissions
        if record.get("person") and record.get("object")
    }


def _matches_any_admission(
    example: Any,
    admission_keys: set[tuple[str, str]],
) -> bool:
    prompt = _example_prompt(example)
    return any(person in prompt and obj in prompt for person, obj in admission_keys)


def _is_prior_fact_example(example: Any) -> bool:
    source = _example_source(example)
    fact_targets = {"color", "owner", "place", "training_data"}
    return (
        source_family(source) in {"qa", "fact", "bridge"}
        and source_target(source) in fact_targets
    )


def _example_prompt(example: Any) -> str:
    return _example_value(example, "prompt")


def _example_source(example: Any) -> str:
    return _example_value(example, "source", "unknown")


def _example_target(example: Any) -> str:
    return _example_value(example, "target")


def _example_value(example: Any, field: str, default: str = "") -> str:
    return str(getattr(example, field, default))
