"""Default paths for admission probe artifacts."""

from __future__ import annotations

from curriculum import DEFAULT_CORPUS_DIR, PROJECT_DIR


DEFAULT_ADMISSIONS = DEFAULT_CORPUS_DIR / "admissions.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "evals" / "admissions.jsonl"
DEFAULT_PARAPHRASE_OUTPUT = PROJECT_DIR / "evals" / "admission_paraphrases.jsonl"
