"""Generate and audit admission probes from the admitted-memory log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .curriculum import DEFAULT_CORPUS_DIR, PROJECT_DIR, read_jsonl


DEFAULT_ADMISSIONS = DEFAULT_CORPUS_DIR / "admissions.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "evals" / "admissions.jsonl"
DEFAULT_PARAPHRASE_OUTPUT = PROJECT_DIR / "evals" / "admission_paraphrases.jsonl"


def admission_probe_records(admissions: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for fact in admissions:
        person = fact["person"]
        obj = fact["object"]
        slug = f"{person}-{obj}"
        place = f"{fact['relation']} the {fact['container']}"
        records.extend(
            [
                {
                    "id": f"admission-place-{slug}",
                    "prompt": f"question: where is {person}'s {obj}?\nanswer:",
                    "target": f" {place}.",
                },
                {
                    "id": f"admission-color-{slug}",
                    "prompt": f"question: what color is {person}'s {obj}?\nanswer:",
                    "target": f" {fact['color']}.",
                },
                {
                    "id": f"admission-owner-{slug}",
                    "prompt": f"question: who has the {obj}?\nanswer:",
                    "target": f" {person}.",
                },
                {
                    "id": f"admission-status-{slug}",
                    "prompt": f"question: is {person}'s {obj} part of your training data?\nanswer:",
                    "target": " yes.",
                },
            ]
        )
    return records


def admission_paraphrase_probe_records(admissions: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for fact in admissions:
        person = fact["person"]
        obj = fact["object"]
        slug = f"{person}-{obj}"
        place = f"{fact['relation']} the {fact['container']}"
        records.extend(
            [
                {
                    "id": f"admission-para-place-tell-{slug}",
                    "prompt": f"tell me the place of {person} {obj}\nanswer:",
                    "target": f" {place}.",
                },
                {
                    "id": f"admission-para-place-ask-{slug}",
                    "prompt": f"ask: place for {person} {obj}\nanswer:",
                    "target": f" {place}.",
                },
                {
                    "id": f"admission-para-color-belongs-{slug}",
                    "prompt": f"which color belongs to {person} {obj}\nanswer:",
                    "target": f" {fact['color']}.",
                },
                {
                    "id": f"admission-para-color-ask-{slug}",
                    "prompt": f"ask: color for {person} {obj}\nanswer:",
                    "target": f" {fact['color']}.",
                },
                {
                    "id": f"admission-para-owner-belongs-{slug}",
                    "prompt": f"which person has {obj}\nanswer:",
                    "target": f" {person}.",
                },
                {
                    "id": f"admission-para-owner-ask-{slug}",
                    "prompt": f"ask: owner for {obj}\nanswer:",
                    "target": f" {person}.",
                },
                {
                    "id": f"admission-para-status-tag-{slug}",
                    "prompt": f"training data: {person} {obj}\nanswer:",
                    "target": " yes.",
                },
            ]
        )
    return records


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def sync_admission_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    records = admission_probe_records(read_jsonl(admissions_path))
    write_jsonl(records, output_path)
    return {
        "admissions": str(admissions_path),
        "output": str(output_path),
        "records": len(records),
    }


def sync_admission_paraphrase_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    output_path: Path = DEFAULT_PARAPHRASE_OUTPUT,
) -> dict[str, Any]:
    records = admission_paraphrase_probe_records(read_jsonl(admissions_path))
    write_jsonl(records, output_path)
    return {
        "admissions": str(admissions_path),
        "output": str(output_path),
        "records": len(records),
    }


def sync_all_admission_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    direct_output_path: Path = DEFAULT_OUTPUT,
    paraphrase_output_path: Path = DEFAULT_PARAPHRASE_OUTPUT,
) -> dict[str, Any]:
    return {
        "direct": sync_admission_probes(admissions_path, direct_output_path),
        "paraphrases": sync_admission_paraphrase_probes(
            admissions_path,
            paraphrase_output_path,
        ),
    }


def audit_admission_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    probes_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    expected = admission_probe_records(read_jsonl(admissions_path))
    actual = read_jsonl(probes_path)
    expected_by_id = {record["id"]: record for record in expected}
    actual_by_id = {record["id"]: record for record in actual}
    missing = sorted(set(expected_by_id) - set(actual_by_id))
    extra = sorted(set(actual_by_id) - set(expected_by_id))
    mismatched = [
        record_id
        for record_id in sorted(set(expected_by_id) & set(actual_by_id))
        if expected_by_id[record_id] != actual_by_id[record_id]
    ]
    return {
        "admissions": str(admissions_path),
        "probes": str(probes_path),
        "expected_records": len(expected),
        "actual_records": len(actual),
        "missing_ids": missing,
        "extra_ids": extra,
        "mismatched_ids": mismatched,
        "passed": not missing and not extra and not mismatched,
    }


def audit_admission_paraphrase_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    probes_path: Path = DEFAULT_PARAPHRASE_OUTPUT,
) -> dict[str, Any]:
    expected = admission_paraphrase_probe_records(read_jsonl(admissions_path))
    actual = read_jsonl(probes_path)
    expected_by_id = {record["id"]: record for record in expected}
    actual_by_id = {record["id"]: record for record in actual}
    missing = sorted(set(expected_by_id) - set(actual_by_id))
    extra = sorted(set(actual_by_id) - set(expected_by_id))
    mismatched = [
        record_id
        for record_id in sorted(set(expected_by_id) & set(actual_by_id))
        if expected_by_id[record_id] != actual_by_id[record_id]
    ]
    return {
        "admissions": str(admissions_path),
        "probes": str(probes_path),
        "expected_records": len(expected),
        "actual_records": len(actual),
        "missing_ids": missing,
        "extra_ids": extra,
        "mismatched_ids": mismatched,
        "passed": not missing and not extra and not mismatched,
    }


def audit_all_admission_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    direct_probes_path: Path = DEFAULT_OUTPUT,
    paraphrase_probes_path: Path = DEFAULT_PARAPHRASE_OUTPUT,
) -> dict[str, Any]:
    direct = audit_admission_probes(admissions_path, direct_probes_path)
    paraphrases = audit_admission_paraphrase_probes(
        admissions_path,
        paraphrase_probes_path,
    )
    return {
        "direct": direct,
        "paraphrases": paraphrases,
        "passed": direct["passed"] and paraphrases["passed"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admissions", type=Path, default=DEFAULT_ADMISSIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paraphrases-output", type=Path, default=DEFAULT_PARAPHRASE_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        result = audit_all_admission_probes(
            args.admissions,
            args.output,
            args.paraphrases_output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    result = sync_all_admission_probes(
        args.admissions,
        args.output,
        args.paraphrases_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
