from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.corpus_hygiene import (
    attach_replay_plan_summary,
    build_corpus_hygiene_report,
    build_training_plan,
    duplicate_values,
    source_mixture,
    train_eval_overlap,
)
from closed_world_lm.candidate_quarantine import (
    build_candidate_quarantine_manifest,
    candidate_quarantine_summary,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class CorpusHygieneTest(unittest.TestCase):
    def test_duplicate_values_reports_duplicate_ids(self) -> None:
        summary = duplicate_values(
            [{"id": "one"}, {"id": "two"}, {"id": "one"}],
            "id",
        )

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["duplicate_count"], 1)
        self.assertEqual(summary["duplicates"][0]["value"], "one")

    def test_source_mixture_tracks_candidate_ratio(self) -> None:
        examples = [
            SimpleNamespace(prompt="p1", target=" a.", source="qa:place"),
            SimpleNamespace(prompt="p2", target=" b.", source="candidate:repair"),
            SimpleNamespace(prompt="p3", target=" c.", source="bridge:color"),
        ]

        mixture = source_mixture(examples)

        self.assertEqual(mixture["by_family"]["candidate"], 1)
        self.assertEqual(mixture["candidate_examples"], 1)
        self.assertAlmostEqual(mixture["candidate_ratio"], 1 / 3)

    def test_train_eval_overlap_marks_protected_prompt_overlap(self) -> None:
        examples = [
            SimpleNamespace(
                prompt="question: hidden?\nanswer:",
                target=" secret.",
                source="qa:place",
            )
        ]
        with tempfile.TemporaryDirectory() as temp:
            heldout_path = Path(temp) / "heldout.jsonl"
            write_jsonl(
                heldout_path,
                [
                    {
                        "id": "heldout-one",
                        "prompt": "question: hidden?\nanswer:",
                        "target": " secret.",
                    }
                ],
            )

            report = train_eval_overlap(
                "question: hidden?\nanswer:\nanswer: secret.",
                examples,
                [heldout_path],
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["protected_prompt_overlap_count"], 1)
        self.assertEqual(report["protected_train_text_prompt_overlap_count"], 1)
        self.assertEqual(
            report["eval_sets"]["heldout"]["protected_prompt_overlap_count"],
            1,
        )

    def test_build_corpus_hygiene_report_summarizes_core_risks(self) -> None:
        examples = [
            SimpleNamespace(prompt="q1", target=" a.", source="qa:place"),
            SimpleNamespace(prompt="q2", target=" b.", source="qa:place"),
            SimpleNamespace(prompt="q3", target=" c.", source="candidate:repair"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus_dir = root / "corpus"
            build_dir = root / "build"
            eval_dir = root / "evals"
            write_json(corpus_dir / "glossary.json", {"entries": [{"word": "a"}]})
            write_json(
                corpus_dir / "grammar.json",
                {
                    "sentence_templates": [{}],
                    "story_facts": [{"id": "story"}],
                    "unknown_facts": [],
                    "self_facts": [],
                    "learning_rules": [],
                },
            )
            write_jsonl(corpus_dir / "admissions.jsonl", [{"id": "admit"}])
            train_text_path = build_dir / "train.txt"
            train_text_path.parent.mkdir(parents=True)
            train_text_path.write_text("q1\nanswer: a.\n", encoding="utf-8")
            eval_path = eval_dir / "qa.jsonl"
            write_jsonl(eval_path, [{"id": "qa-one", "prompt": "q1", "target": " a."}])

            report = build_corpus_hygiene_report(
                "test-component",
                corpus_dir,
                train_text_path,
                [eval_path],
                examples,
            )

        self.assertEqual(report["kind"], "corpus_hygiene_report")
        self.assertEqual(report["corpus_sources"]["admitted_facts"], 1)
        self.assertEqual(report["candidate_ratio"], 1 / 3)
        self.assertEqual(
            report["train_eval_overlap"]["eval_sets"]["qa"]["prompt_overlap_count"],
            1,
        )

    def test_training_plan_records_boundary_and_replay_summary(self) -> None:
        examples = [
            SimpleNamespace(prompt="q1", target=" a.", source="qa:place"),
            SimpleNamespace(prompt="q2", target=" b.", source="candidate:repair"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_candidate_quarantine_manifest(
                "transformer-answer-train",
                "run-001",
            )
            plan = build_training_plan(
                "transformer-answer-train",
                "run-001",
                root / "build" / "train.txt",
                root / "corpus",
                [root / "evals" / "qa.jsonl"],
                examples,
                [examples[0], examples[0], examples[1]],
                root / "run" / "corpus_hygiene.json",
                replay_plan_path=root / "run" / "direct_answer_replay_plan.json",
                candidate_quarantine_path=root / "run" / "candidate_quarantine.json",
                candidate_quarantine_summary=candidate_quarantine_summary(manifest),
            )
            updated = attach_replay_plan_summary(
                plan,
                {
                    "profile_aware_targets": True,
                    "branch_count": 2,
                    "replay_count": 2,
                    "profiles": {
                        "qa:place": {"missing_target_count": 1},
                        "qa:color": {"missing_target_count": 0},
                    },
                },
                root / "run" / "direct_answer_replay_plan.json",
            )

        self.assertFalse(plan["data_boundary"]["pretrained_weights"])
        self.assertEqual(plan["candidate_policy"]["candidate_examples"], 1)
        self.assertEqual(
            plan["candidate_policy"]["status"],
            "training_examples_contain_candidates",
        )
        self.assertFalse(plan["candidate_policy"]["candidate_records_are_training_data"])
        self.assertEqual(
            plan["candidate_policy"]["candidate_quarantine"]["summary"]["candidate_count"],
            0,
        )
        self.assertEqual(updated["replay_plan"]["status"], "written")
        self.assertEqual(updated["replay_plan"]["profiles_with_missing_targets"], ["qa:place"])


if __name__ == "__main__":
    unittest.main()
