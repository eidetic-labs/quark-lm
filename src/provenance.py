"""Corpus provenance snapshots and diffs for closed-world training runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from curriculum import read_json, read_jsonl


PROJECT_DIR = Path(__file__).resolve().parents[1]


def source_path(corpus_dir: Path, ledger_path: str) -> Path:
    path = Path(ledger_path)
    parts = path.parts
    if parts and parts[0] == "corpus":
        return corpus_dir.joinpath(*parts[1:])
    return PROJECT_DIR / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int | None:
    if path.suffix != ".jsonl" or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def corpus_snapshot(corpus_dir: Path) -> dict[str, Any]:
    ledger = read_json(corpus_dir / "ledger.json")
    source_files: dict[str, Any] = {}
    for source in ledger["sources"]:
        path = source_path(corpus_dir, source["path"])
        exists = path.exists()
        source_files[source["id"]] = {
            "path": str(path),
            "ledger_path": source["path"],
            "kind": source["kind"],
            "allowed_for_curriculum_generation": source["allowed_for_curriculum_generation"],
            "allowed_for_training": source["allowed_for_training"],
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
            "bytes": path.stat().st_size if exists else None,
            "jsonl_records": jsonl_count(path),
        }

    admissions_path = corpus_dir / "admissions.jsonl"
    admissions = read_jsonl(admissions_path)
    return {
        "schema_version": 1,
        "ledger_version": ledger["version"],
        "corpus_dir": str(corpus_dir),
        "source_files": source_files,
        "admissions": {
            "path": str(admissions_path),
            "count": len(admissions),
            "ids": [record["id"] for record in admissions],
        },
    }


def diff_corpus_snapshots(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    previous_report_path: Path | None = None,
) -> dict[str, Any]:
    if previous is None:
        return {
            "mode": "corpus_snapshot",
            "compare_report": str(previous_report_path) if previous_report_path else None,
            "status": "not_evaluated_no_previous_snapshot",
            "source_files": {},
            "admissions": {
                "added": current["admissions"]["ids"],
                "removed": [],
                "unchanged": [],
            },
        }

    source_diff = diff_source_files(current["source_files"], previous.get("source_files", {}))
    current_ids = set(current["admissions"]["ids"])
    previous_ids = set(previous.get("admissions", {}).get("ids", []))
    return {
        "mode": "corpus_snapshot",
        "compare_report": str(previous_report_path) if previous_report_path else None,
        "status": "evaluated",
        "source_files": source_diff,
        "admissions": {
            "added": sorted(current_ids - previous_ids),
            "removed": sorted(previous_ids - current_ids),
            "unchanged": sorted(current_ids & previous_ids),
            "previous_count": len(previous_ids),
            "current_count": len(current_ids),
        },
    }


def diff_source_files(
    current_sources: dict[str, Any],
    previous_sources: dict[str, Any],
) -> dict[str, Any]:
    source_diff: dict[str, Any] = {}
    for source_id in sorted(set(current_sources) | set(previous_sources)):
        if source_id not in previous_sources:
            status = "added"
        elif source_id not in current_sources:
            status = "removed"
        elif current_sources[source_id].get("sha256") != previous_sources[source_id].get("sha256"):
            status = "changed"
        else:
            status = "unchanged"
        source_diff[source_id] = {
            "status": status,
            "previous_sha256": previous_sources.get(source_id, {}).get("sha256"),
            "current_sha256": current_sources.get(source_id, {}).get("sha256"),
            "previous_records": previous_sources.get(source_id, {}).get("jsonl_records"),
            "current_records": current_sources.get(source_id, {}).get("jsonl_records"),
        }
    return source_diff


def corpus_diff_for_report(
    current_snapshot: dict[str, Any],
    previous_report_path: Path | None,
) -> dict[str, Any]:
    if previous_report_path is None:
        return diff_corpus_snapshots(current_snapshot, None)
    with previous_report_path.open("r", encoding="utf-8") as handle:
        previous_report = json.load(handle)
    return diff_corpus_snapshots(
        current_snapshot,
        previous_report.get("corpus_snapshot"),
        previous_report_path,
    )
