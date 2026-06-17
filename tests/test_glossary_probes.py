from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curriculum import read_json, read_jsonl
from glossary_probes import (
    audit_glossary_probes,
    glossary_probe_records,
    sync_glossary_probes,
)


class GlossaryProbesTest(unittest.TestCase):
    def test_checked_in_glossary_probes_match_glossary(self) -> None:
        audit = audit_glossary_probes(
            ROOT / "corpus" / "glossary.json",
            ROOT / "evals" / "glossary.jsonl",
        )

        self.assertTrue(audit["passed"])
        glossary = read_json(ROOT / "corpus" / "glossary.json")
        self.assertEqual(audit["expected_records"], len(glossary["probe_words"]) * 2)

    def test_records_are_generated_from_probe_words(self) -> None:
        glossary = {
            "probe_words": ["corpus"],
            "entries": [
                {
                    "word": "corpus",
                    "definition": "the admitted training data",
                }
            ],
        }

        records = glossary_probe_records(glossary)

        self.assertEqual(
            records,
            [
                {
                    "id": "glossary-meaning-corpus",
                    "prompt": "question: what does corpus mean?\nanswer:",
                    "target": " the admitted training data.",
                },
                {
                    "id": "glossary-define-corpus",
                    "prompt": "define corpus\nanswer:",
                    "target": " the admitted training data.",
                },
            ],
        )

    def test_sync_and_audit_glossary_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            glossary = root / "glossary.json"
            probes = root / "glossary-probes.jsonl"
            glossary.write_text(
                '{"probe_words": ["stone"], "entries": '
                '[{"word": "stone", "definition": "a small object from the ground"}]}\n',
                encoding="utf-8",
            )

            result = sync_glossary_probes(glossary, probes)

            self.assertEqual(result["records"], 2)
            self.assertEqual(len(read_jsonl(probes)), 2)
            self.assertTrue(audit_glossary_probes(glossary, probes)["passed"])

    def test_probe_words_must_exist_in_entries(self) -> None:
        glossary = read_json(ROOT / "corpus" / "glossary.json")
        bad = {**glossary, "probe_words": ["missingword"]}

        with self.assertRaises(ValueError):
            glossary_probe_records(bad)


if __name__ == "__main__":
    unittest.main()
