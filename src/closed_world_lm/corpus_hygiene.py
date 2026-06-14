"""Corpus hygiene and training-plan artifacts for closed-world runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def corpus_source_summary(corpus_dir: Path) -> dict[str, Any]:
    glossary = read_json(corpus_dir / "glossary.json")
    grammar = read_json(corpus_dir / "grammar.json")
    admissions = read_jsonl(corpus_dir / "admissions.jsonl")
    return {
        "corpus_dir": str(corpus_dir),
        "glossary_entries": len(glossary.get("entries", [])),
        "sentence_templates": len(grammar.get("sentence_templates", [])),
        "story_facts": len(grammar.get("story_facts", [])),
        "admitted_facts": len(admissions),
        "unknown_facts": len(grammar.get("unknown_facts", [])),
        "unknown_owner_objects": len(grammar.get("unknown_owner_objects", [])),
        "self_facts": len(grammar.get("self_facts", [])),
        "learning_rules": len(grammar.get("learning_rules", [])),
    }


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


def eval_set_counts(eval_paths: list[Path]) -> dict[str, Any]:
    return {
        path.stem: {
            "path": str(path),
            "records": len(read_jsonl(path)),
        }
        for path in eval_paths
    }


def build_training_plan(
    component: str,
    run_id: str,
    train_text_path: Path,
    corpus_dir: Path,
    eval_paths: list[Path],
    training_examples: list[Any],
    training_pool: list[Any],
    hygiene_path: Path,
    planned_artifacts: list[Path] | None = None,
    replay_plan_path: Path | None = None,
    candidate_quarantine_path: Path | None = None,
    candidate_quarantine_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_examples = source_mixture(training_examples)["candidate_examples"]
    candidate_status = "candidate_quarantine_missing"
    if candidate_examples > 0:
        candidate_status = "training_examples_contain_candidates"
    elif candidate_quarantine_summary is not None:
        if candidate_quarantine_summary.get("candidate_count", 0) == 0:
            candidate_status = "candidate_quarantine_empty"
        elif candidate_quarantine_summary.get("not_training_eligible_count", 0) > 0:
            candidate_status = "candidate_quarantine_holds_candidates"
        else:
            candidate_status = "candidate_quarantine_all_candidates_admitted"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "training_plan",
        "component": component,
        "run_id": run_id,
        "allowed_data_sources": [
            str(corpus_dir / "glossary.json"),
            str(corpus_dir / "grammar.json"),
            str(corpus_dir / "admissions.jsonl"),
            str(train_text_path),
            *[str(path) for path in eval_paths],
        ],
        "data_boundary": {
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "unledgered_training_text": False,
        },
        "hygiene_report": str(hygiene_path),
        "eval_sets": eval_set_counts(eval_paths),
        "examples": {
            "base_examples": len(training_examples),
            "scheduled_examples": len(training_pool),
            "base_source_mixture": source_mixture(training_examples),
            "scheduled_source_mixture": source_mixture(training_pool),
            "rare_profile_coverage": rare_profile_coverage(training_examples),
        },
        "candidate_policy": {
            "candidate_examples": candidate_examples,
            "candidate_ratio": (
                candidate_examples / len(training_examples)
                if training_examples
                else 0.0
            ),
            "status": candidate_status,
            "candidate_records_are_training_data": False,
            "rule": "Candidate records are excluded from training until admitted into the ledgered corpus and converted into curriculum lessons.",
            "candidate_quarantine": {
                "path": (
                    str(candidate_quarantine_path)
                    if candidate_quarantine_path is not None
                    else None
                ),
                "summary": candidate_quarantine_summary,
            },
        },
        "replay_plan": {
            "status": "planned" if replay_plan_path is not None else "not_applicable",
            "path": str(replay_plan_path) if replay_plan_path is not None else None,
        },
        "planned_artifacts": [
            str(path) for path in (planned_artifacts or [])
        ],
    }


def attach_replay_plan_summary(
    training_plan: dict[str, Any],
    replay_plan: dict[str, Any],
    replay_plan_path: Path,
) -> dict[str, Any]:
    updated = dict(training_plan)
    profiles = replay_plan.get("profiles", {})
    updated["replay_plan"] = {
        "status": "written",
        "path": str(replay_plan_path),
        "profile_aware_targets": replay_plan.get("profile_aware_targets"),
        "branch_count": replay_plan.get("branch_count"),
        "replay_count": replay_plan.get("replay_count"),
        "profile_count": len(profiles),
        "profiles_with_missing_targets": [
            profile
            for profile, summary in sorted(profiles.items())
            if summary.get("missing_target_count", 0) > 0
        ],
    }
    return updated
