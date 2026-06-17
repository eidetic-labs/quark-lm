"""Admission probe drift audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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


ProbeBuilder = Callable[[list[dict[str, Any]]], list[dict[str, str]]]


def _audit_probe_file(
    admissions_path: Path,
    probes_path: Path,
    build_expected: ProbeBuilder,
) -> dict[str, Any]:
    expected = build_expected(read_jsonl(admissions_path))
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


def audit_admission_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    probes_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    return _audit_probe_file(
        admissions_path,
        probes_path,
        admission_probe_records,
    )


def audit_admission_paraphrase_probes(
    admissions_path: Path = DEFAULT_ADMISSIONS,
    probes_path: Path = DEFAULT_PARAPHRASE_OUTPUT,
) -> dict[str, Any]:
    return _audit_probe_file(
        admissions_path,
        probes_path,
        admission_paraphrase_probe_records,
    )


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
