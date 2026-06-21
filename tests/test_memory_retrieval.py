from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_cards import build_memory_cards
from memory_index import ClosedWorldMemoryIndex as MemoryIndexImplementation
from memory_retrieval import (
    ClosedWorldMemoryIndex,
    build_retrieval_memory_report,
    write_retrieval_memory_report,
)
from memory_retrieval_signatures import prompt_signature, signatures_match, tokenize


class MemoryRetrievalTest(unittest.TestCase):
    def test_memory_retrieval_facade_exports_index_implementation(self) -> None:
        self.assertIs(ClosedWorldMemoryIndex, MemoryIndexImplementation)

    def test_prompt_signatures_cover_retrieval_prompt_families(self) -> None:
        place = prompt_signature("tell me the place of ivy map\nanswer:")
        owner = prompt_signature("question: who has the map?\nanswer:")
        glossary = prompt_signature("define corpus\nanswer:")

        self.assertEqual(
            place,
            {"intent": "place", "person": "ivy", "object": "map"},
        )
        self.assertEqual(owner, {"intent": "owner", "object": "map"})
        self.assertEqual(glossary, {"intent": "glossary", "word": "corpus"})
        self.assertTrue(
            signatures_match(
                {"intent": "place", "person": "ivy"},
                {"intent": "place", "person": "ivy", "object": "map"},
            )
        )
        self.assertEqual(tokenize("Ivy's map, v2!"), ["ivy", "s", "map", "v2"])

    def test_memory_cards_are_built_from_corpus_profiles(self) -> None:
        cards = build_memory_cards(ROOT / "corpus")
        profiles = {card.profile for card in cards}

        self.assertIn("owner", profiles)
        self.assertIn("self", profiles)
        self.assertIn("learning", profiles)
        self.assertIn("glossary", profiles)

    def test_closed_world_memory_answers_owner_and_paraphrase_without_weights(self) -> None:
        index = ClosedWorldMemoryIndex.from_corpus(ROOT / "corpus")

        owner = index.answer_prompt("question: who has the ball?\nanswer:")
        paraphrase = index.answer_prompt("tell me the place of mia ball\nanswer:")
        unknown = index.answer_prompt("question: who has the water?\nanswer:")
        withheld = index.answer_prompt("question: who has the map?\nanswer:")

        self.assertEqual(owner["answer"], " mia.")
        self.assertEqual(owner["memory_card"]["source"], "corpus:grammar:story_facts")
        self.assertEqual(paraphrase["answer"], " under the box.")
        self.assertFalse(unknown["retrieved"])
        self.assertEqual(unknown["answer"], " unknown.")
        # ivy-map is withheld: not retrievable from memory either (consistent withhold).
        self.assertFalse(withheld["retrieved"])
        self.assertEqual(withheld["answer"], " unknown.")

    def test_retrieval_report_is_corpus_only_and_tracks_consolidation_surface(self) -> None:
        report = build_retrieval_memory_report(ROOT / "corpus")

        self.assertEqual(report["kind"], "retrieval_memory_report")
        self.assertFalse(report["dataset_exclusivity"]["uses_external_model"])
        self.assertFalse(report["dataset_exclusivity"]["external_embeddings"])
        self.assertFalse(report["dataset_exclusivity"]["updates_weights"])
        self.assertGreater(report["memory"]["card_count"], 0)
        self.assertEqual(report["evals"]["owner"]["exact_rate"], 1.0)
        self.assertEqual(report["evals"]["paraphrases"]["exact_rate"], 1.0)
        self.assertEqual(report["evals"]["admission_paraphrases"]["exact_rate"], 1.0)
        self.assertEqual(
            report["self_improvement"]["status"],
            "memory_serves_before_weight_consolidation",
        )

    def test_retrieval_report_can_be_written_as_run_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "retrieval_memory_report.json"
            report = build_retrieval_memory_report(ROOT / "corpus")

            write_retrieval_memory_report(path, report)

            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["summary"]["exact_rate"], report["summary"]["exact_rate"])


if __name__ == "__main__":
    unittest.main()
