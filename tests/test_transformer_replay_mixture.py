from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from answer_model import AnswerExample
from transformer_replay_mixture import (
    build_transformer_replay_mixture_report,
    replay_mixture_summary,
    write_transformer_replay_mixture_report,
)


class TransformerReplayMixtureTest(unittest.TestCase):
    def test_replay_mixture_reports_required_buckets(self) -> None:
        examples = [
            AnswerExample("fact place teacher tree\nanswer:", " near the garden.", "qa:place"),
            AnswerExample("fact place mia ball\nanswer:", " under the box.", "qa:place"),
            AnswerExample("question: where is noah's ball?\nanswer:", " unknown.", "unknown:place"),
            AnswerExample("fact self dataset\nanswer:", " the admitted corpus.", "fact:self"),
            AnswerExample("define corpus\nanswer:", " the admitted training data.", "qa:glossary"),
        ]
        eval_records = {
            "heldout": [{"prompt": "h", "target": " near the garden."}],
            "paraphrases": [{"prompt": "p", "target": " under the box."}],
            "unknowns": [{"prompt": "u", "target": " unknown."}],
        }

        report = build_transformer_replay_mixture_report(
            run_id="run-001",
            train_text_path=Path("build/train.txt"),
            examples=examples,
            training_pool=[*examples, examples[0]],
            eval_records=eval_records,
            admissions=[{"person": "teacher", "object": "tree"}],
        )

        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["buckets"]["new_lessons"]["count"], 1)
        self.assertEqual(report["buckets"]["prior_accepted_facts"]["count"], 1)
        self.assertEqual(report["buckets"]["glossary_self_facts"]["count"], 2)
        self.assertGreater(report["buckets"]["unknown_policy_probes"]["total_count"], 0)
        self.assertEqual(report["buckets"]["heldout_paraphrases"]["count"], 2)
        self.assertGreater(report["buckets"]["tokenizer_stress_strings"]["count"], 0)

    def test_replay_mixture_summary_is_json_artifact_safe(self) -> None:
        report = build_transformer_replay_mixture_report(
            run_id="run-001",
            train_text_path=Path("build/train.txt"),
            examples=[
                AnswerExample("fact place teacher tree\nanswer:", " near the garden.", "qa:place"),
                AnswerExample("fact place mia ball\nanswer:", " under the box.", "qa:place"),
                AnswerExample("question: where is noah's ball?\nanswer:", " unknown.", "unknown:place"),
                AnswerExample("fact self dataset\nanswer:", " the admitted corpus.", "fact:self"),
                AnswerExample("define corpus\nanswer:", " the admitted training data.", "qa:glossary"),
            ],
            training_pool=[],
            eval_records={
                "heldout": [{"target": " near the garden."}],
                "paraphrases": [{"target": " under the box."}],
                "unknowns": [{"target": " unknown."}],
            },
            admissions=[{"person": "teacher", "object": "tree"}],
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "replay_mixture_report.json"
            write_transformer_replay_mixture_report(path, report)
            written = json.loads(path.read_text(encoding="utf-8"))

        summary = replay_mixture_summary(written)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["bucket_count"], 6)


if __name__ == "__main__":
    unittest.main()
