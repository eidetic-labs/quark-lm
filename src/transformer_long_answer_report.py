"""Durable long-answer diagnostic reports for answer-training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neural_char_metrics import continuation_nll
from tokenizer_protocol import TokenizerProtocol
from transformer_answer_diagnostics import answer_diagnostics


LONG_ANSWER_REPORT_KIND = "transformer_long_answer_diagnostics"
LONG_ANSWER_REPORT_SCHEMA_VERSION = 1
DEFAULT_RECORDS_PER_EVAL = 1
TOP_CANDIDATE_LIMIT = 5


def build_long_answer_diagnostics_report(
    *,
    run_id: str,
    model: Any,
    tokenizer: TokenizerProtocol,
    eval_records: dict[str, list[dict[str, Any]]],
    eval_candidates: dict[str, list[str]],
    generation_config: Any,
    train_time_seconds: float,
    direct_answer_metrics: dict[str, Any] | None,
    records_per_eval: int = DEFAULT_RECORDS_PER_EVAL,
) -> dict[str, Any]:
    records = []
    for eval_name, selected in _selected_long_records(
        eval_records,
        tokenizer,
        records_per_eval,
    ):
        candidates = eval_candidates.get(eval_name, [])
        for record in selected:
            diagnostics = answer_diagnostics(
                model,
                tokenizer,
                record["prompt"],
                record["target"],
                generation_config,
            )
            diagnostics.update(
                {
                    "eval": eval_name,
                    "id": record["id"],
                    "candidate_ranking": _candidate_ranking(
                        model,
                        tokenizer,
                        record["prompt"],
                        record["target"],
                        candidates,
                    ),
                }
            )
            records.append(diagnostics)
    return {
        "schema_version": LONG_ANSWER_REPORT_SCHEMA_VERSION,
        "kind": LONG_ANSWER_REPORT_KIND,
        "component": "transformer-answer-train",
        "run_id": run_id,
        "records_per_eval": records_per_eval,
        "records": records,
        "summary": _summary(records, train_time_seconds, direct_answer_metrics),
    }


def write_long_answer_diagnostics_report(
    path: Path,
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _selected_long_records(
    eval_records: dict[str, list[dict[str, Any]]],
    tokenizer: TokenizerProtocol,
    records_per_eval: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    selected = []
    for eval_name, records in sorted(eval_records.items()):
        ranked = sorted(
            records,
            key=lambda record: (
                len(tokenizer.encode(record["target"])),
                len(record["target"]),
                record["id"],
            ),
            reverse=True,
        )
        selected.append((eval_name, ranked[:records_per_eval]))
    return selected


def _candidate_ranking(
    model: Any,
    tokenizer: TokenizerProtocol,
    prompt: str,
    target: str,
    candidates: list[str],
) -> dict[str, Any]:
    candidate_set = list(dict.fromkeys([*candidates, target]))
    scores = [
        {
            "target": candidate,
            "target_nll": continuation_nll(model, tokenizer, prompt, candidate),
        }
        for candidate in candidate_set
    ]
    ranked = sorted(scores, key=lambda item: float(item["target_nll"]))
    target_rank = next(
        index
        for index, item in enumerate(ranked, start=1)
        if item["target"] == target
    )
    return {
        "candidate_count": len(ranked),
        "target_rank": target_rank,
        "predicted_candidate": ranked[0]["target"] if ranked else None,
        "top_candidates": ranked[:TOP_CANDIDATE_LIMIT],
    }


def _summary(
    records: list[dict[str, Any]],
    train_time_seconds: float,
    direct_answer_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    exact = sum(1 for record in records if record["exact_match"])
    generation_times = [record["generation_time_ms"] for record in records]
    target_counts = [record["target_token_count"] for record in records]
    return {
        "record_count": len(records),
        "exact": exact,
        "exact_rate": exact / len(records) if records else 0.0,
        "max_target_token_count": max(target_counts, default=0),
        "avg_generation_time_ms": (
            sum(generation_times) / len(generation_times) if generation_times else 0.0
        ),
        "train_time_seconds": train_time_seconds,
        "branch_diversity_target": _branch_diversity_target(direct_answer_metrics),
    }


def _branch_diversity_target(
    direct_answer_metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not direct_answer_metrics:
        return None
    final = direct_answer_metrics.get("final", {})
    return final.get("branch_diversity_target")
