"""Paths and eval-set constants for the closed-world answer model."""

from __future__ import annotations

from pathlib import Path

from curriculum import DEFAULT_CORPUS_DIR, DEFAULT_OUTPUT_DIR


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TEXT = DEFAULT_OUTPUT_DIR / "train.txt"
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "answer-latest"
DEFAULT_EVALS = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "paraphrases.jsonl",
    PROJECT_DIR / "evals" / "owner.jsonl",
    PROJECT_DIR / "evals" / "self.jsonl",
    PROJECT_DIR / "evals" / "learning.jsonl",
    PROJECT_DIR / "evals" / "admissions.jsonl",
    PROJECT_DIR / "evals" / "admission_paraphrases.jsonl",
    PROJECT_DIR / "evals" / "glossary.jsonl",
]
