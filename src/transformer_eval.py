"""Transformer evaluation scoring and report artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from answer_candidates import answer_type_for, menu_for
from neural_char_metrics import continuation_nll
from probes import read_jsonl, summarize
from transformer_model import GenerationConfig


def load_probe_records(probe_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    return {path.stem: read_jsonl(path) for path in probe_paths}


def eval_candidates_from_records(
    probe_records: dict[str, list[dict[str, Any]]],
) -> list[str]:
    return sorted(
        {
            record["target"]
            for records in probe_records.values()
            for record in records
        }
    )


def score_transformer_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_new_chars: int,
    generation_config: GenerationConfig,
    candidates: list[str] | None = None,
    menus: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for record in records:
        generation = model.generate_with_trace(
            tokenizer,
            record["prompt"],
            max_new_chars,
            generation_config,
        )
        # Per-type menu (de-contaminated) when provided; else the legacy global pool.
        record_candidates = (
            menu_for(record["prompt"], menus) if menus is not None else candidates
        )
        candidate_scores = []
        predicted_candidate = None
        if record_candidates is not None:
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
                for candidate in record_candidates
            ]
            predicted_candidate = min(
                candidate_scores,
                key=lambda item: float(item["target_nll"]),
            )["target"]
        target = record["target"]
        scored.append(
            {
                "id": record["id"],
                "prompt": record["prompt"],
                "target": target,
                "answer_type": answer_type_for(record["prompt"]),
                "completion": generation["text"],
                "generation_trace": generation["trace"],
                "generation_cache": generation["cache"],
                "exact_match": generation["text"][: len(target)] == target,
                "candidate_match": predicted_candidate == target
                if predicted_candidate is not None
                else None,
                "predicted_candidate": predicted_candidate,
                "candidate_scores": candidate_scores,
                "target_nll": continuation_nll(
                    model,
                    tokenizer,
                    record["prompt"],
                    target,
                ),
            }
        )
    return scored


def score_transformer_evals(
    model: Any,
    tokenizer: Any,
    probe_records: dict[str, list[dict[str, Any]]],
    max_new_chars: int,
    generation_config: GenerationConfig,
    candidates: list[str] | None = None,
    menus: dict[str, list[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: score_transformer_records(
            model,
            tokenizer,
            records,
            max_new_chars,
            generation_config,
            candidates=candidates,
            menus=menus,
        )
        for name, records in sorted(probe_records.items())
    }


def build_transformer_eval_report(
    checkpoint: Path,
    probe_paths: list[Path],
    probe_records: dict[str, list[dict[str, Any]]],
    scored_by_eval: dict[str, list[dict[str, Any]]],
    candidates: list[str],
    generation_config: GenerationConfig,
    samples_jsonl: Path | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "candidate_count": len(candidates),
        "generation_config": asdict(generation_config),
        "eval_manifest": {
            "probe_paths": [str(path) for path in probe_paths],
            "probe_counts": {
                name: len(records)
                for name, records in sorted(probe_records.items())
            },
            "samples_jsonl": str(samples_jsonl) if samples_jsonl else None,
        },
        "evals": {
            name: summarize(records)
            for name, records in sorted(scored_by_eval.items())
        },
    }


def write_eval_samples(
    path: Path,
    scored_by_eval: dict[str, list[dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, records in sorted(scored_by_eval.items()):
            for record in records:
                handle.write(json.dumps({"eval": name, **record}, sort_keys=True) + "\n")


def write_eval_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
