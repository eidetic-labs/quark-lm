"""Responder evaluation summaries for self-improvement reports."""

from __future__ import annotations

from typing import Any

from answer_model import DEFAULT_EVALS
from corpus_responder import CorpusResponder
from probes import read_jsonl


def summarize_exact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": result["count"],
        "exact": result["exact"],
        "exact_rate": result["exact_rate"],
    }


def evaluate_responder(train_text: str) -> dict[str, Any]:
    responder = CorpusResponder.train_from_text(train_text)
    return {
        path.stem: summarize_exact(responder.evaluate(read_jsonl(path)))
        for path in DEFAULT_EVALS
    }
