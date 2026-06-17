"""Filesystem defaults for transformer commands."""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "transformer-latest"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "transformer.json"
DEFAULT_PROBES = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "paraphrases.jsonl",
]
