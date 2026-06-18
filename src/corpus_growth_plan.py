"""Preflight reports for staged closed-world corpus growth."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from admission_probe_records import (
    admission_paraphrase_probe_records,
    admission_probe_records,
)
from admit import admitted_fact_from_record
from corpus_artifacts import SCHEMA_VERSION, read_jsonl, write_json_artifact
from corpus_duplicate_summary import duplicate_values
from curriculum import DEFAULT_CORPUS_DIR, PROJECT_DIR


DEFAULT_OUTPUT = PROJECT_DIR / "build" / "corpus_growth_plan.json"
DEFAULT_EVALS = [
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "owner.jsonl",
    PROJECT_DIR / "evals" / "admissions.jsonl",
    PROJECT_DIR / "evals" / "admission_paraphrases.jsonl",
]


def build_corpus_growth_plan(
    *,
    batch_path: Path,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    eval_paths: list[Path] | None = None,
    retention_limit: int = 8,
    stress_limit: int = 8,
) -> dict[str, Any]:
    eval_paths = eval_paths or DEFAULT_EVALS
    batch_records = read_jsonl(batch_path)
    facts = [admitted_fact_from_record(record) for record in batch_records]
    fact_records = [asdict(fact) for fact in facts]
    admissions_path = corpus_dir / "admissions.jsonl"
    existing_admissions = read_jsonl(admissions_path)
    direct_probes = admission_probe_records(fact_records)
    paraphrase_probes = admission_paraphrase_probe_records(fact_records)
    duplicate_checks = _duplicate_checks(batch_records, existing_admissions)
    split_checks = _split_checks([*direct_probes, *paraphrase_probes], eval_paths)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "corpus_growth_plan",
        "component": "corpus-growth",
        "batch": {
            "path": str(batch_path),
            "records": len(fact_records),
            "ids": [fact.id for fact in facts],
        },
        "source_provenance": {
            "corpus_dir": str(corpus_dir),
            "admissions": str(admissions_path),
            "eval_paths": [str(path) for path in eval_paths],
            "batch_file": str(batch_path),
        },
        "duplicate_checks": duplicate_checks,
        "train_eval_split_checks": split_checks,
        "retention_probes": _retention_probe_summary(
            existing_admissions,
            retention_limit,
        ),
        "unknown_policy_probes": _unknown_policy_probes(facts),
        "tokenizer_stress_examples": _tokenizer_stress_examples(facts, stress_limit),
        "generated_probe_counts": {
            "direct": len(direct_probes),
            "paraphrase": len(paraphrase_probes),
        },
    }
    report["passed"] = duplicate_checks["passed"] and split_checks["passed"]
    report["status"] = "ready_for_admission" if report["passed"] else "blocked"
    return report


def write_corpus_growth_plan(path: Path, report: dict[str, Any]) -> None:
    write_json_artifact(path, report)


def _duplicate_checks(
    batch_records: list[dict[str, Any]],
    existing_admissions: list[dict[str, Any]],
) -> dict[str, Any]:
    batch_ids = duplicate_values(batch_records, "id")
    existing_ids = {str(record.get("id")) for record in existing_admissions}
    conflicts = sorted(
        {
            str(record.get("id"))
            for record in batch_records
            if str(record.get("id")) in existing_ids
        }
    )
    fact_keys = [
        {"fact_key": _fact_key(record)}
        for record in [*existing_admissions, *batch_records]
    ]
    fact_key_duplicates = duplicate_values(fact_keys, "fact_key")
    return {
        "batch_ids": batch_ids,
        "existing_id_conflicts": conflicts,
        "fact_keys": fact_key_duplicates,
        "passed": (
            batch_ids["passed"]
            and not conflicts
            and fact_key_duplicates["duplicate_count"] == 0
        ),
    }


def _split_checks(
    generated_probes: list[dict[str, str]],
    eval_paths: list[Path],
) -> dict[str, Any]:
    eval_prompts = {}
    for path in eval_paths:
        for record in read_jsonl(path):
            eval_prompts.setdefault(str(record.get("prompt", "")), []).append(
                {"eval": path.stem, "id": record.get("id")}
            )
    overlaps = [
        {
            "probe_id": probe["id"],
            "prompt": probe["prompt"],
            "eval_records": eval_prompts[probe["prompt"]],
        }
        for probe in generated_probes
        if probe["prompt"] in eval_prompts
    ]
    return {
        "generated_probe_count": len(generated_probes),
        "prompt_overlap_count": len(overlaps),
        "overlaps": overlaps[:20],
        "passed": not overlaps,
    }


def _retention_probe_summary(
    existing_admissions: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    records = admission_probe_records(existing_admissions[:limit])
    return {
        "source": "existing_admissions",
        "admission_count": min(len(existing_admissions), limit),
        "probe_count": len(records),
        "sample_ids": [record["id"] for record in records[:limit]],
        "rule": "Corpus growth batches must retain prior accepted facts.",
    }


def _unknown_policy_probes(facts: list[Any]) -> list[dict[str, str]]:
    return [
        {
            "id": f"unknown-after-{fact.id}",
            "prompt": f"question: where is outside {fact.object}?\nanswer:",
            "target": " unknown.",
        }
        for fact in facts
    ]


def _tokenizer_stress_examples(facts: list[Any], limit: int) -> list[dict[str, Any]]:
    examples = []
    for fact in facts:
        place = f"{fact.relation} the {fact.container}"
        text = f"question: where is {fact.person}'s {fact.object}? answer: {place}."
        examples.append({"id": fact.id, "text": text, "chars": len(text)})
    return sorted(examples, key=lambda item: item["chars"], reverse=True)[:limit]


def _fact_key(record: dict[str, Any]) -> str:
    return f"{record.get('person')}::{record.get('object')}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--eval", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_corpus_growth_plan(
        batch_path=args.batch,
        corpus_dir=args.corpus_dir,
        eval_paths=args.eval,
    )
    write_corpus_growth_plan(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
