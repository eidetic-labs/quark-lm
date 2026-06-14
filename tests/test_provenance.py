from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.curriculum import read_json, read_jsonl
from closed_world_lm.provenance import corpus_snapshot, diff_corpus_snapshots


def admission_ids() -> list[str]:
    return [record["id"] for record in read_jsonl(ROOT / "corpus" / "admissions.jsonl")]


class ProvenanceTest(unittest.TestCase):
    def test_corpus_snapshot_records_ledger_sources_and_admissions(self) -> None:
        snapshot = corpus_snapshot(ROOT / "corpus")

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertIn("admissions-v0", snapshot["source_files"])
        self.assertTrue(snapshot["source_files"]["admissions-v0"]["allowed_for_training"])
        self.assertEqual(
            snapshot["source_files"]["admissions-v0"]["jsonl_records"],
            len(admission_ids()),
        )
        self.assertIn("admission-paraphrase-probes-v0", snapshot["source_files"])
        self.assertFalse(
            snapshot["source_files"]["admission-paraphrase-probes-v0"]["allowed_for_training"]
        )
        self.assertEqual(
            snapshot["source_files"]["admission-paraphrase-probes-v0"]["jsonl_records"],
            len(admission_ids()) * 7,
        )
        self.assertIn("glossary-probes-v0", snapshot["source_files"])
        self.assertFalse(snapshot["source_files"]["glossary-probes-v0"]["allowed_for_training"])
        glossary = read_json(ROOT / "corpus" / "glossary.json")
        self.assertEqual(
            snapshot["source_files"]["glossary-probes-v0"]["jsonl_records"],
            len(glossary["probe_words"]) * 2,
        )
        self.assertEqual(snapshot["admissions"]["ids"], admission_ids())

    def test_diff_corpus_snapshots_tracks_added_admissions_and_changed_sources(self) -> None:
        current = corpus_snapshot(ROOT / "corpus")
        previous = deepcopy(current)
        previous["admissions"]["ids"] = admission_ids()[:-1]
        previous["source_files"]["admissions-v0"]["sha256"] = "old"
        previous["source_files"]["admissions-v0"]["jsonl_records"] = len(admission_ids()) - 1

        diff = diff_corpus_snapshots(current, previous)

        self.assertEqual(diff["status"], "evaluated")
        self.assertEqual(diff["admissions"]["added"], admission_ids()[-1:])
        self.assertEqual(diff["admissions"]["previous_count"], len(admission_ids()) - 1)
        self.assertEqual(diff["admissions"]["current_count"], len(admission_ids()))
        self.assertEqual(diff["source_files"]["admissions-v0"]["status"], "changed")


if __name__ == "__main__":
    unittest.main()
