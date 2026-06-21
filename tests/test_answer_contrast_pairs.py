"""Corpus-driven contrast pairs are entity-swapped, leakage-safe, withheld-excluded."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_contrast_pairs import build_contrast_pairs

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = REPO_ROOT / "corpus" / "grammar.json"


class ContrastPairsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = json.loads(GRAMMAR.read_text(encoding="utf-8"))
        self.pairs = build_contrast_pairs(self.grammar)

    def test_pairs_contrast_a_fact_against_entity_swapped_unknown(self) -> None:
        self.assertTrue(self.pairs)
        for in_example, ooc_example in self.pairs:
            self.assertNotEqual(in_example.target, " unknown.")  # owner -> concrete
            self.assertEqual(ooc_example.target, " unknown.")     # non-owner -> abstain
            self.assertNotEqual(in_example.prompt, ooc_example.prompt)  # entity swapped

    def test_withheld_facts_are_not_used_for_contrast(self) -> None:
        withheld_ids = set(self.grammar.get("withheld_fact_ids", []))
        withheld = [
            (f["person"], f["object"])
            for f in self.grammar["story_facts"]
            if f["id"] in withheld_ids
        ]
        in_prompts = " ".join(in_example.prompt for in_example, _ in self.pairs)
        for person, obj in withheld:
            self.assertNotIn(f"{person}'s {obj}", in_prompts)

    def test_ooc_targets_are_not_corpus_facts(self) -> None:
        # The non-owner pairing must be neither a real fact nor a declared unknown,
        # so a contrast OOC example can never coincide with an eval probe.
        excluded = {(f["person"], f["object"]) for f in self.grammar["story_facts"]}
        excluded |= {
            (f["person"], f["object"]) for f in self.grammar.get("unknown_facts", [])
        }
        for _in_example, ooc_example in self.pairs:
            # prompt form: "question: where is P's O?" / "what color is P's O?"
            body = ooc_example.prompt.split("is ", 1)[1]
            person, rest = body.split("'s ", 1)
            obj = rest.split("?", 1)[0]
            self.assertNotIn((person, obj), excluded)


if __name__ == "__main__":
    unittest.main()
