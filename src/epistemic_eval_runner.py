"""Standalone runner: score a checkpoint against eval probes, emit the report.

Turns the manual recovery demo into a repeatable epistemic dashboard. Reuses the
existing scoring path (score_transformer_evals -> summarize) and feeds it to the
epistemic spine (epistemic_report). Self-contained: run with

    PYTHONPATH=src python -m epistemic_eval_runner \
        --checkpoint runs/<run>/transformer_answer.json \
        --probe evals/qa.jsonl --probe evals/heldout.jsonl \
        --probe evals/unknowns.jsonl [--train-text build/train.txt] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from answer_candidates import candidates_by_type
from corpus_responder import CorpusResponder
from epistemic_report import epistemic_report
from probes import summarize
from transformer_eval import (
    eval_candidates_from_records,
    load_probe_records,
    score_transformer_evals,
)
from transformer_model import GenerationConfig
from transformer_tiny_lm import TinyTransformerLM


def run_epistemic_eval(
    *,
    model: Any,
    tokenizer: Any,
    probe_paths: list[Path],
    max_new_chars: int = 64,
    generation_config: GenerationConfig | None = None,
    responder: Any | None = None,
) -> dict[str, Any]:
    """Score the probes with an in-memory model and return the epistemic report."""

    probe_records = load_probe_records(probe_paths)
    # De-contaminated per-type menus from the corpus answer space when a responder
    # is available; otherwise fall back to the legacy global candidate pool.
    menus = (
        candidates_by_type(responder)
        if isinstance(responder, CorpusResponder)
        else None
    )
    candidates = None if menus is not None else eval_candidates_from_records(probe_records)
    scored_by_set = score_transformer_evals(
        model,
        tokenizer,
        probe_records,
        max_new_chars,
        generation_config or GenerationConfig(),
        candidates=candidates,
        menus=menus,
    )
    evals = {name: summarize(records) for name, records in scored_by_set.items()}
    return epistemic_report(evals, scored_by_set, tokenizer.vocab_size, responder)


def run_from_checkpoint(
    checkpoint: str | Path,
    probe_paths: list[str | Path],
    *,
    max_new_chars: int = 64,
    train_text: str | Path | None = None,
) -> dict[str, Any]:
    """Load a checkpoint, build the corpus oracle if available, run the report."""

    model, tokenizer = TinyTransformerLM.load(Path(checkpoint))
    responder = None
    try:
        responder = (
            CorpusResponder.load_train_text(Path(train_text))
            if train_text is not None
            else CorpusResponder.load_train_text()
        )
    except Exception:
        responder = None  # oracle is optional; report still computes the rest
    return run_epistemic_eval(
        model=model,
        tokenizer=tokenizer,
        probe_paths=[Path(path) for path in probe_paths],
        max_new_chars=max_new_chars,
        responder=responder,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Closed-world epistemic evaluation report.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", action="append", dest="probes", required=True)
    parser.add_argument("--train-text", default=None)
    parser.add_argument("--max-new-chars", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report = run_from_checkpoint(
        args.checkpoint,
        args.probes,
        max_new_chars=args.max_new_chars,
        train_text=args.train_text,
    )
    print(json.dumps(report["headline"], indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
