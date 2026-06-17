"""Admit a new closed-world fact before it can be used for training."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from admission_probes import (
    DEFAULT_OUTPUT as DEFAULT_PROBES,
    DEFAULT_PARAPHRASE_OUTPUT as DEFAULT_PARAPHRASE_PROBES,
    sync_admission_paraphrase_probes,
    sync_admission_probes,
)
from curriculum import DEFAULT_CORPUS_DIR, read_jsonl


WORD_RE = re.compile(r"^[a-z]+$")
ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
DEFAULT_ADMISSIONS = DEFAULT_CORPUS_DIR / "admissions.jsonl"
REQUIRED_FIELDS = ("id", "person", "object", "color", "relation", "container")


@dataclass(frozen=True)
class AdmittedFact:
    id: str
    person: str
    object: str
    color: str
    relation: str
    container: str
    admission: str = "I learned something new, and now it is part of my training data."


def validate_word(name: str, value: str) -> str:
    if not WORD_RE.match(value):
        raise ValueError(f"{name} must contain lowercase letters only")
    return value


def validate_id(value: str) -> str:
    if not ID_RE.match(value):
        raise ValueError("id must start with a lowercase letter and contain only lowercase letters, digits, and hyphens")
    return value


def admitted_fact_from_record(record: dict[str, Any]) -> AdmittedFact:
    missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
    if missing:
        raise ValueError(f"missing admission fields: {', '.join(missing)}")
    return AdmittedFact(
        id=validate_id(str(record["id"])),
        person=validate_word("person", str(record["person"])),
        object=validate_word("object", str(record["object"])),
        color=validate_word("color", str(record["color"])),
        relation=validate_word("relation", str(record["relation"])),
        container=validate_word("container", str(record["container"])),
        admission=str(record.get("admission", AdmittedFact.admission)),
    )


def admitted_fact_from_args(args: argparse.Namespace) -> AdmittedFact:
    return admitted_fact_from_record(
        {field: getattr(args, field) for field in REQUIRED_FIELDS}
    )


def read_admission_batch(path: Path) -> list[AdmittedFact]:
    return [admitted_fact_from_record(record) for record in read_jsonl(path)]


def append_admissions(path: Path, facts: list[AdmittedFact]) -> dict[str, Any]:
    if not facts:
        raise ValueError("admission batch is empty")
    records = read_jsonl(path)
    existing_ids = {record["id"] for record in records}
    batch_ids = [fact.id for fact in facts]
    duplicate_existing = sorted(existing_ids.intersection(batch_ids))
    duplicate_batch = sorted(
        {fact_id for fact_id in batch_ids if batch_ids.count(fact_id) > 1}
    )
    if duplicate_existing:
        raise ValueError(f"admission id already exists: {', '.join(duplicate_existing)}")
    if duplicate_batch:
        raise ValueError(f"duplicate admission id in batch: {', '.join(duplicate_batch)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for fact in facts:
            handle.write(json.dumps(asdict(fact), sort_keys=True) + "\n")
    return {
        "admitted": [asdict(fact) for fact in facts],
        "admitted_count": len(facts),
        "training_status": "admitted_pending_weight_update",
        "next_step": "run self_improve answer-cycle",
    }


def append_admission(path: Path, fact: AdmittedFact) -> dict[str, Any]:
    batch_result = append_admissions(path, [fact])
    return {
        **batch_result,
        "admitted": asdict(fact),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_ADMISSIONS)
    parser.add_argument("--probes", type=Path, default=None)
    parser.add_argument("--paraphrase-probes", type=Path, default=None)
    parser.add_argument("--no-sync-probes", action="store_true")
    parser.add_argument("--batch", type=Path, default=None)
    parser.add_argument("--id")
    parser.add_argument("--person")
    parser.add_argument("--object")
    parser.add_argument("--color")
    parser.add_argument("--relation")
    parser.add_argument("--container")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch:
        result = append_admissions(args.path, read_admission_batch(args.batch))
    else:
        result = append_admission(args.path, admitted_fact_from_args(args))
    probes_path = args.probes
    paraphrase_probes_path = args.paraphrase_probes
    if args.path == DEFAULT_ADMISSIONS and not args.no_sync_probes:
        probes_path = probes_path or DEFAULT_PROBES
        paraphrase_probes_path = paraphrase_probes_path or DEFAULT_PARAPHRASE_PROBES
    if not args.no_sync_probes:
        probe_sync: dict[str, Any] = {}
        if probes_path is not None:
            probe_sync["direct"] = sync_admission_probes(args.path, probes_path)
        if paraphrase_probes_path is not None:
            probe_sync["paraphrases"] = sync_admission_paraphrase_probes(
                args.path,
                paraphrase_probes_path,
            )
        if probe_sync:
            result["probe_sync"] = probe_sync
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
