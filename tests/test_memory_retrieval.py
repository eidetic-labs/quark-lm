from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.memory_retrieval import (
    ClosedWorldMemoryIndex,
    build_retrieval_memory_report,
    write_retrieval_memory_report,
)


class MemoryRetrievalTest(unittest.TestCase):
    def test_closed_world_memory_answers_owner_and_paraphrase_without_weights(self) -> None:
        index = ClosedWorldMemoryIndex.from_corpus(ROOT / "corpus")

        owner = index.answer_prompt("question: who has the map?\nanswer:")
        paraphrase = index.answer_prompt("tell me the place of ivy map\nanswer:")
        unknown = index.answer_prompt("question: who has the water?\nanswer:")

        self.assertEqual(owner["answer"], " ivy.")
        self.assertEqual(owner["memory_card"]["source"], "corpus:grammar:story_facts")
        self.assertEqual(paraphrase["answer"], " on the shelf.")
        self.assertFalse(unknown["retrieved"])
        self.assertEqual(unknown["answer"], " unknown.")

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
