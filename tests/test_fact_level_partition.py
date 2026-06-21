"""Fact-level held-out partition guard (Phase 3, Unit B).

A withheld fact must be fully absent from the training corpus (never admitted, so
not learnable), and every eval probe about a withheld fact must expect the
closed-world oracle's answer ( " unknown." for place/color/owner, " no." for a
training-data question) -- a genuine fact-level boundary, not phrasing-held-out.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import support  # noqa: F401  (inserts src/ onto sys.path)
from curriculum import build_curriculum
from corpus_responder import (
    COLOR_ASK_RE,
    COLOR_BELONGS_RE,
    COLOR_QUESTION_RE,
    CorpusResponder,
    OWNER_ASK_RE,
    OWNER_BELONGS_RE,
    OWNER_QUESTION_RE,
    PLACE_ASK_RE,
    PLACE_TELL_RE,
    TRAINING_DATA_QUESTION_RE,
    TRAINING_DATA_TAG_RE,
    WHERE_QUESTION_RE,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_TEXT = REPO_ROOT / "build" / "train.txt"
GRAMMAR = REPO_ROOT / "corpus" / "grammar.json"
EVAL_DIR = REPO_ROOT / "evals"

_PAIR_PATTERNS = [
    WHERE_QUESTION_RE, COLOR_QUESTION_RE, PLACE_ASK_RE, COLOR_ASK_RE,
    PLACE_TELL_RE, COLOR_BELONGS_RE, TRAINING_DATA_QUESTION_RE, TRAINING_DATA_TAG_RE,
]
_OBJECT_PATTERNS = [OWNER_QUESTION_RE, OWNER_ASK_RE, OWNER_BELONGS_RE]


def _withheld_keys(grammar: dict) -> set[tuple[str, str]]:
    ids = set(grammar.get("withheld_fact_ids", []))
    return {(f["person"], f["object"]) for f in grammar["story_facts"] if f["id"] in ids}


def _about_withheld(prompt: str, pairs: set[tuple[str, str]], objects: set[str]) -> bool:
    for pattern in _PAIR_PATTERNS:
        match = pattern.search(prompt)
        if match and (match["person"], match["object"]) in pairs:
            return True
    for pattern in _OBJECT_PATTERNS:
        match = pattern.search(prompt)
        if match and match["object"] in objects:
            return True
    return False


class FactLevelPartitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = json.loads(GRAMMAR.read_text(encoding="utf-8"))
        self.withheld = _withheld_keys(self.grammar)
        self.objects = {obj for _person, obj in self.withheld}

    def test_partition_is_declared(self) -> None:
        self.assertTrue(self.withheld, "withheld_fact_ids must define a non-empty partition")

    def test_withheld_facts_not_learnable_from_corpus(self) -> None:
        # Build the corpus in-memory from the grammar so this guard runs in CI even
        # though build/train.txt is a gitignored artifact.
        responder = CorpusResponder.train_from_text(build_curriculum(seed=3).train_text)
        for pair in sorted(self.withheld):
            self.assertNotIn(pair, responder.facts, f"{pair} leaked into the training corpus")

    def test_withheld_eval_records_expect_oracle_answer(self) -> None:
        if not EVAL_DIR.exists():
            self.skipTest("evals not present")
        responder = CorpusResponder.train_from_text(build_curriculum(seed=3).train_text)
        mismatches: list[tuple[str, str, str, str]] = []
        for path in sorted(EVAL_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if _about_withheld(record["prompt"], self.withheld, self.objects):
                    expected = responder.answer_prompt(record["prompt"])
                    if record["target"] != expected:
                        mismatches.append((path.name, record["id"], record["target"], expected))
        self.assertEqual(mismatches, [], f"withheld probes inconsistent with the oracle: {mismatches[:8]}")


if __name__ == "__main__":
    unittest.main()
