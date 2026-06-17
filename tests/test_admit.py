from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admit import AdmittedFact, append_admission, append_admissions, main
from curriculum import read_jsonl


class AdmitTest(unittest.TestCase):
    def test_append_admission_records_pending_weight_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "admissions.jsonl"
            fact = AdmittedFact(
                id="learned-child-book",
                person="child",
                object="book",
                color="blue",
                relation="on",
                container="table",
            )

            result = append_admission(path, fact)

            self.assertEqual(result["training_status"], "admitted_pending_weight_update")
            self.assertEqual(read_jsonl(path)[0]["id"], "learned-child-book")

    def test_duplicate_admission_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "admissions.jsonl"
            fact = AdmittedFact(
                id="learned-child-book",
                person="child",
                object="book",
                color="blue",
                relation="on",
                container="table",
            )
            append_admission(path, fact)

            with self.assertRaises(ValueError):
                append_admission(path, fact)

    def test_append_admissions_records_a_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "admissions.jsonl"
            facts = [
                AdmittedFact(
                    id="learned-child-book",
                    person="child",
                    object="book",
                    color="blue",
                    relation="on",
                    container="table",
                ),
                AdmittedFact(
                    id="learned-teacher-bag",
                    person="teacher",
                    object="bag",
                    color="yellow",
                    relation="near",
                    container="shelf",
                ),
            ]

            result = append_admissions(path, facts)

            self.assertEqual(result["admitted_count"], 2)
            self.assertEqual(
                [record["id"] for record in read_jsonl(path)],
                ["learned-child-book", "learned-teacher-bag"],
            )

    def test_duplicate_batch_id_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "admissions.jsonl"
            facts = [
                AdmittedFact(
                    id="learned-child-book",
                    person="child",
                    object="book",
                    color="blue",
                    relation="on",
                    container="table",
                ),
                AdmittedFact(
                    id="learned-child-book",
                    person="teacher",
                    object="bag",
                    color="yellow",
                    relation="near",
                    container="shelf",
                ),
            ]

            with self.assertRaises(ValueError):
                append_admissions(path, facts)

            self.assertEqual(read_jsonl(path), [])

    def test_cli_can_sync_generated_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            admissions = root / "admissions.jsonl"
            probes = root / "admissions-probes.jsonl"
            paraphrase_probes = root / "admission-paraphrases.jsonl"

            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--path",
                        str(admissions),
                        "--probes",
                        str(probes),
                        "--paraphrase-probes",
                        str(paraphrase_probes),
                        "--id",
                        "learned-child-book",
                        "--person",
                        "child",
                        "--object",
                        "book",
                        "--color",
                        "blue",
                        "--relation",
                        "on",
                        "--container",
                        "table",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(len(read_jsonl(probes)), 4)
            self.assertEqual(len(read_jsonl(paraphrase_probes)), 7)


if __name__ == "__main__":
    unittest.main()
