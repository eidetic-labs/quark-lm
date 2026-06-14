from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.admission_probes import (
    admission_paraphrase_probe_records,
    admission_probe_records,
    audit_admission_paraphrase_probes,
    audit_all_admission_probes,
    audit_admission_probes,
    sync_all_admission_probes,
    sync_admission_paraphrase_probes,
    sync_admission_probes,
)
from closed_world_lm.curriculum import read_jsonl


def current_admission_count() -> int:
    return len(read_jsonl(ROOT / "corpus" / "admissions.jsonl"))


class AdmissionProbesTest(unittest.TestCase):
    def test_checked_in_admission_probes_match_admitted_memory_log(self) -> None:
        audit = audit_all_admission_probes(
            ROOT / "corpus" / "admissions.jsonl",
            ROOT / "evals" / "admissions.jsonl",
            ROOT / "evals" / "admission_paraphrases.jsonl",
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["direct"]["expected_records"], current_admission_count() * 4)
        self.assertEqual(audit["paraphrases"]["expected_records"], current_admission_count() * 7)

    def test_records_are_generated_from_admitted_facts(self) -> None:
        records = admission_probe_records(
            [
                {
                    "id": "learned-child-bag",
                    "person": "child",
                    "object": "bag",
                    "color": "yellow",
                    "relation": "near",
                    "container": "shelf",
                }
            ]
        )

        self.assertEqual(len(records), 4)
        self.assertIn(
            {
                "id": "admission-place-child-bag",
                "prompt": "question: where is child's bag?\nanswer:",
                "target": " near the shelf.",
            },
            records,
        )
        self.assertIn(
            {
                "id": "admission-status-child-bag",
                "prompt": "question: is child's bag part of your training data?\nanswer:",
                "target": " yes.",
            },
            records,
        )

    def test_paraphrase_records_are_generated_from_admitted_facts(self) -> None:
        records = admission_paraphrase_probe_records(
            [
                {
                    "id": "learned-child-bag",
                    "person": "child",
                    "object": "bag",
                    "color": "yellow",
                    "relation": "near",
                    "container": "shelf",
                }
            ]
        )

        self.assertEqual(len(records), 7)
        self.assertIn(
            {
                "id": "admission-para-place-tell-child-bag",
                "prompt": "tell me the place of child bag\nanswer:",
                "target": " near the shelf.",
            },
            records,
        )
        self.assertIn(
            {
                "id": "admission-para-status-tag-child-bag",
                "prompt": "training data: child bag\nanswer:",
                "target": " yes.",
            },
            records,
        )

    def test_sync_and_audit_admission_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            admissions = root / "admissions.jsonl"
            probes = root / "admissions-probes.jsonl"
            paraphrases = root / "admission-paraphrases.jsonl"
            admissions.write_text(
                '{"id": "learned-child-bag", "person": "child", "object": "bag", '
                '"color": "yellow", "relation": "near", "container": "shelf"}\n',
                encoding="utf-8",
            )

            result = sync_admission_probes(admissions, probes)
            paraphrase_result = sync_admission_paraphrase_probes(admissions, paraphrases)
            all_result = sync_all_admission_probes(admissions, probes, paraphrases)

            self.assertEqual(result["records"], 4)
            self.assertEqual(paraphrase_result["records"], 7)
            self.assertEqual(all_result["direct"]["records"], 4)
            self.assertEqual(all_result["paraphrases"]["records"], 7)
            self.assertEqual(len(read_jsonl(probes)), 4)
            self.assertEqual(len(read_jsonl(paraphrases)), 7)
            self.assertTrue(audit_admission_probes(admissions, probes)["passed"])
            self.assertTrue(audit_admission_paraphrase_probes(admissions, paraphrases)["passed"])
            self.assertTrue(audit_all_admission_probes(admissions, probes, paraphrases)["passed"])

    def test_audit_detects_probe_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            admissions = root / "admissions.jsonl"
            probes = root / "admissions-probes.jsonl"
            admissions.write_text(
                '{"id": "learned-child-bag", "person": "child", "object": "bag", '
                '"color": "yellow", "relation": "near", "container": "shelf"}\n',
                encoding="utf-8",
            )
            probes.write_text(
                '{"id": "admission-place-child-bag", "prompt": "bad", "target": "bad"}\n',
                encoding="utf-8",
            )

            audit = audit_admission_probes(admissions, probes)

            self.assertFalse(audit["passed"])
            self.assertIn("admission-color-child-bag", audit["missing_ids"])
            self.assertIn("admission-place-child-bag", audit["mismatched_ids"])


if __name__ == "__main__":
    unittest.main()
