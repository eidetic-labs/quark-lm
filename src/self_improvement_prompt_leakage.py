"""Prompt leakage audits for self-improvement lessons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from probes import read_jsonl


PROJECT_DIR = Path(__file__).resolve().parents[1]


def audit_prompt_leakage(
    lesson_paths: list[Path],
    eval_path: Path,
    protected_id_contains: str | None = None,
) -> dict[str, Any]:
    records = read_jsonl(eval_path)
    if protected_id_contains is not None:
        records = [record for record in records if protected_id_contains in record["id"]]
    eval_prompts = {record["prompt"] for record in records}
    leaked: list[dict[str, str]] = []
    for lesson_path in lesson_paths:
        if not lesson_path.exists():
            leaked.append(
                {
                    "lesson_source": str(lesson_path),
                    "prompt": "<missing lesson file>",
                }
            )
            continue
        with lesson_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("prompt") in eval_prompts:
                    leaked.append(
                        {
                            "lesson_source": str(lesson_path),
                            "prompt": record["prompt"],
                        }
                    )
    return {
        "eval_source": str(eval_path),
        "protected_id_contains": protected_id_contains,
        "lesson_sources": [str(path) for path in lesson_paths],
        "leaked_prompts": leaked,
        "passed": not leaked,
    }


def audit_all_protected_prompts(lesson_paths: list[Path]) -> dict[str, Any]:
    return {
        "heldout": audit_prompt_leakage(
            lesson_paths,
            PROJECT_DIR / "evals" / "heldout.jsonl",
        ),
        "owner_heldout": audit_prompt_leakage(
            lesson_paths,
            PROJECT_DIR / "evals" / "owner.jsonl",
            protected_id_contains="-heldout-",
        ),
    }
