from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from corpus_growth_plan import build_corpus_growth_plan, write_corpus_growth_plan


class CorpusGrowthPlanTest(unittest.TestCase):
    def test_growth_plan_records_clean_batch_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus_dir = root / "corpus"
            batch = root / "batch.jsonl"
            eval_path = root / "eval.jsonl"
            _write_jsonl(
                corpus_dir / "admissions.jsonl",
                [
                    _fact("learned-mia-cup", "mia", "cup"),
                ],
            )
            _write_jsonl(batch, [_fact("learned-lee-book", "lee", "book")])
            _write_jsonl(eval_path, [{"id": "heldout", "prompt": "x", "target": " y."}])

            report = build_corpus_growth_plan(
                batch_path=batch,
                corpus_dir=corpus_dir,
                eval_paths=[eval_path],
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "ready_for_admission")
        self.assertEqual(report["generated_probe_counts"]["direct"], 4)
        self.assertGreater(report["retention_probes"]["probe_count"], 0)
        self.assertEqual(report["unknown_policy_probes"][0]["target"], " unknown.")
        self.assertEqual(report["tokenizer_stress_examples"][0]["id"], "learned-lee-book")

    def test_growth_plan_blocks_duplicate_and_eval_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus_dir = root / "corpus"
            batch = root / "batch.jsonl"
            eval_path = root / "eval.jsonl"
            existing = _fact("learned-mia-cup", "mia", "cup")
            _write_jsonl(corpus_dir / "admissions.jsonl", [existing])
            _write_jsonl(batch, [existing])
            _write_jsonl(
                eval_path,
                [
                    {
                        "id": "overlap",
                        "prompt": "question: where is mia's cup?\nanswer:",
                        "target": " on the table.",
                    }
                ],
            )

            report = build_corpus_growth_plan(
                batch_path=batch,
                corpus_dir=corpus_dir,
                eval_paths=[eval_path],
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(
            report["duplicate_checks"]["existing_id_conflicts"],
            ["learned-mia-cup"],
        )
        self.assertEqual(
            report["train_eval_split_checks"]["prompt_overlap_count"],
            1,
        )

    def test_growth_plan_writer_serializes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "growth.json"
            write_corpus_growth_plan(path, {"kind": "corpus_growth_plan"})
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["kind"], "corpus_growth_plan")


def _fact(record_id: str, person: str, obj: str) -> dict[str, str]:
    return {
        "id": record_id,
        "person": person,
        "object": obj,
        "color": "blue",
        "relation": "on",
        "container": "table",
    }


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
