"""Probe scoring helpers for closed-world learning runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neural_char_model import CharMLP, continuation_nll
from tokenizer import CharTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl") from exc
    return records


def score_records(
    model: CharMLP,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    max_new_chars: int,
    candidates: list[str] | None = None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for record in records:
        target = record["target"]
        completion = model.generate(tokenizer, record["prompt"], max_new_chars=max_new_chars)
        candidate_scores = []
        predicted_candidate = None
        if candidates is not None:
            candidate_scores = [
                {
                    "target": candidate,
                    "target_nll": continuation_nll(
                        model,
                        tokenizer,
                        record["prompt"],
                        candidate,
                    ),
                }
                for candidate in candidates
            ]
            predicted_candidate = min(
                candidate_scores,
                key=lambda item: float(item["target_nll"]),
            )["target"]
        scored.append(
            {
                "id": record["id"],
                "target": target,
                "completion": completion,
                "exact_match": completion[: len(target)] == target,
                "candidate_match": predicted_candidate == target
                if predicted_candidate is not None
                else None,
                "predicted_candidate": predicted_candidate,
                "target_nll": continuation_nll(
                    model,
                    tokenizer,
                    record["prompt"],
                    target,
                ),
            }
        )
    return scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        return {
            "count": 0,
            "exact": 0,
            "exact_rate": 0.0,
            "candidate": 0,
            "candidate_rate": 0.0,
            "avg_target_nll": 0.0,
        }
    exact = sum(1 for item in scored if item["exact_match"])
    candidate_scored = [item for item in scored if item.get("candidate_match") is not None]
    candidate = sum(1 for item in candidate_scored if item["candidate_match"])
    avg_nll = sum(float(item["target_nll"]) for item in scored) / len(scored)
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored),
        "candidate": candidate,
        "candidate_rate": candidate / len(candidate_scored) if candidate_scored else 0.0,
        "avg_target_nll": avg_nll,
    }


def target_nll_summary(
    model: CharMLP,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        return {"count": 0, "avg_target_nll": 0.0}
    total = 0.0
    for record in records:
        total += continuation_nll(model, tokenizer, record["prompt"], record["target"])
    return {"count": len(records), "avg_target_nll": total / len(records)}
