"""Corpus source-file summaries for hygiene reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_artifacts import read_json, read_jsonl


def corpus_source_summary(corpus_dir: Path) -> dict[str, Any]:
    glossary = read_json(corpus_dir / "glossary.json")
    grammar = read_json(corpus_dir / "grammar.json")
    admissions = read_jsonl(corpus_dir / "admissions.jsonl")
    return {
        "corpus_dir": str(corpus_dir),
        "glossary_entries": len(glossary.get("entries", [])),
        "sentence_templates": len(grammar.get("sentence_templates", [])),
        "story_facts": len(grammar.get("story_facts", [])),
        "admitted_facts": len(admissions),
        "unknown_facts": len(grammar.get("unknown_facts", [])),
        "unknown_owner_objects": len(grammar.get("unknown_owner_objects", [])),
        "self_facts": len(grammar.get("self_facts", [])),
        "learning_rules": len(grammar.get("learning_rules", [])),
    }
