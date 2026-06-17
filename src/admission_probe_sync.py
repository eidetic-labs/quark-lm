"""Admission probe file synchronization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from admission_probe_paths import (
    DEFAULT_ADMISSIONS,
    DEFAULT_OUTPUT,
    DEFAULT_PARAPHRASE_OUTPUT,
)
from admission_probe_records import (
    admission_paraphrase_probe_records,
    admission_probe_records,
)
from curriculum import read_jsonl


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
