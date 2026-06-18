"""Guarded acceptance for incremental transformer checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from probes import summarize
from transformer_eval import score_transformer_records
from transformer_model import GenerationConfig


def guarded_incremental_update(
    *,
    model_cls: Any,
    base_checkpoint: Path,
    candidate_checkpoint: Path,
    accepted_checkpoint: Path,
    new_lesson_records: list[dict[str, Any]],
    regression_records: list[dict[str, Any]],
    nll_tolerance: float = 0.0,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    generation_config = generation_config or GenerationConfig()
    base_model, base_tokenizer = model_cls.load(base_checkpoint)
    candidate_model, candidate_tokenizer = model_cls.load(candidate_checkpoint)
    if base_tokenizer is None:
        raise ValueError("base checkpoint does not contain a tokenizer")
    if candidate_tokenizer is None:
        raise ValueError("candidate checkpoint does not contain a tokenizer")

    tokenizer_gate = _gate(
        "append_only_tokenizer_extension",
        candidate_tokenizer.extends(base_tokenizer),
        {
            "base_vocab_size": base_tokenizer.vocab_size,
            "candidate_vocab_size": candidate_tokenizer.vocab_size,
        },
    )
    if not tokenizer_gate["passed"]:
        return _decision_report(
            base_checkpoint,
            candidate_checkpoint,
            accepted_checkpoint,
            [tokenizer_gate],
            new_lesson={},
            regression={},
        )

    max_new_chars = _max_target_chars([*new_lesson_records, *regression_records])
    new_scores = score_transformer_records(
        candidate_model,
        candidate_tokenizer,
        new_lesson_records,
        max_new_chars,
        generation_config,
    )
    base_regression_scores = score_transformer_records(
        base_model,
        base_tokenizer,
        regression_records,
        max_new_chars,
        generation_config,
    )
    candidate_regression_scores = score_transformer_records(
        candidate_model,
        candidate_tokenizer,
        regression_records,
        max_new_chars,
        generation_config,
    )
    new_summary = summarize(new_scores)
    base_regression = summarize(base_regression_scores)
    candidate_regression = summarize(candidate_regression_scores)
    gates = [
        tokenizer_gate,
        _gate("new_lesson_exact", new_summary["exact_rate"] == 1.0, new_summary),
        _gate(
            "regression_exact_preserved",
            candidate_regression["exact_rate"] >= base_regression["exact_rate"],
            {
                "base_exact_rate": base_regression["exact_rate"],
                "candidate_exact_rate": candidate_regression["exact_rate"],
            },
        ),
        _gate(
            "regression_target_nll_preserved",
            candidate_regression["avg_target_nll"]
            <= base_regression["avg_target_nll"] + nll_tolerance,
            {
                "base_avg_target_nll": base_regression["avg_target_nll"],
                "candidate_avg_target_nll": candidate_regression["avg_target_nll"],
                "tolerance": nll_tolerance,
            },
        ),
    ]
    report = _decision_report(
        base_checkpoint,
        candidate_checkpoint,
        accepted_checkpoint,
        gates,
        new_lesson={"summary": new_summary, "records": new_scores},
        regression={
            "baseline": {"summary": base_regression, "records": base_regression_scores},
            "candidate": {
                "summary": candidate_regression,
                "records": candidate_regression_scores,
            },
        },
    )
    if report["accepted"]:
        accepted_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_checkpoint, accepted_checkpoint)
    return report


def _decision_report(
    base_checkpoint: Path,
    candidate_checkpoint: Path,
    accepted_checkpoint: Path,
    gates: list[dict[str, Any]],
    *,
    new_lesson: dict[str, Any],
    regression: dict[str, Any],
) -> dict[str, Any]:
    accepted = all(gate["passed"] for gate in gates)
    return {
        "accepted": accepted,
        "base_checkpoint": str(base_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "accepted_checkpoint": str(accepted_checkpoint) if accepted else None,
        "rejection_reasons": [
            gate["name"]
            for gate in gates
            if not gate["passed"]
        ],
        "gates": gates,
        "new_lesson": new_lesson,
        "regression": regression,
    }


def _gate(name: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "required": True,
        "evidence": evidence,
    }


def _max_target_chars(records: list[dict[str, Any]]) -> int:
    return max((len(record["target"]) for record in records), default=1)
