"""Generate and audit glossary definition probes from the nursery glossary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .curriculum import DEFAULT_CORPUS_DIR, PROJECT_DIR, read_json, read_jsonl


DEFAULT_GLOSSARY = DEFAULT_CORPUS_DIR / "glossary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "evals" / "glossary.jsonl"


def glossary_definitions(glossary: dict[str, Any]) -> dict[str, str]:
    return {entry["word"]: entry["definition"] for entry in glossary["entries"]}


def probe_words(glossary: dict[str, Any]) -> list[str]:
    words = list(glossary.get("probe_words", []))
    if not words:
        raise ValueError("glossary probe_words must contain at least one word")
    definitions = glossary_definitions(glossary)
    missing = [word for word in words if word not in definitions]
    if missing:
        raise ValueError(f"glossary probe_words missing from entries: {', '.join(missing)}")
    return words


def glossary_probe_records(glossary: dict[str, Any]) -> list[dict[str, str]]:
    definitions = glossary_definitions(glossary)
    records: list[dict[str, str]] = []
    for word in probe_words(glossary):
        target = f" {definitions[word]}."
        records.extend(
            [
                {
                    "id": f"glossary-meaning-{word}",
                    "prompt": f"question: what does {word} mean?\nanswer:",
                    "target": target,
                },
                {
                    "id": f"glossary-define-{word}",
                    "prompt": f"define {word}\nanswer:",
                    "target": target,
                },
            ]
        )
    return records


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def sync_glossary_probes(
    glossary_path: Path = DEFAULT_GLOSSARY,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    records = glossary_probe_records(read_json(glossary_path))
    write_jsonl(records, output_path)
    return {
        "glossary": str(glossary_path),
        "output": str(output_path),
        "records": len(records),
    }


def audit_glossary_probes(
    glossary_path: Path = DEFAULT_GLOSSARY,
    probes_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    expected = glossary_probe_records(read_json(glossary_path))
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
        "glossary": str(glossary_path),
        "probes": str(probes_path),
        "expected_records": len(expected),
        "actual_records": len(actual),
        "missing_ids": missing,
        "extra_ids": extra,
        "mismatched_ids": mismatched,
        "passed": not missing and not extra and not mismatched,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        result = audit_glossary_probes(args.glossary, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    result = sync_glossary_probes(args.glossary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
