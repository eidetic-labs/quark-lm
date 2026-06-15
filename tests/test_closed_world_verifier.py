from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.candidate_quarantine import (
    build_candidate_quarantine_manifest,
    candidate_quarantine_summary,
    candidate_record,
)
from closed_world_lm.closed_world_verifier import (
    attach_verifier_summary,
    verifier_report_summary,
    verify_candidate_quarantine_manifest,
    verify_candidate_record,
    verify_training_plan,
    write_verifier_report,
)
from closed_world_lm.corpus_hygiene import build_training_plan
from closed_world_lm.respond import CorpusResponder


class ClosedWorldVerifierTest(unittest.TestCase):
    def test_candidate_record_requires_exact_responder_agreement(self) -> None:
        responder = CorpusResponder.train_from_text("fact: ada's orb is shelf.\n")
        record = candidate_record(
            "lesson",
            "generated-probe",
            prompt="question: where is ada's orb?\nanswer:",
            target=" shelf.",
        )

        report = verify_candidate_record(record, responder=responder)

        self.assertTrue(report["passed"])
        self.assertEqual(report["subject_kind"], "candidate_record")
        self.assertFalse(report["uses_external_model"])

    def test_candidate_record_rejects_wrong_closed_world_target(self) -> None:
        responder = CorpusResponder.train_from_text("fact: ada's orb is shelf.\n")
        record = candidate_record(
            "probe",
            "generated-probe",
            prompt="question: where is ada's orb?\nanswer:",
            target=" drawer.",
        )

        report = verify_candidate_record(record, responder=responder)

        self.assertFalse(report["passed"])
        self.assertIn("exact_answer_consistency", report["failed_checks"])

    def test_candidate_quarantine_rejects_training_eligible_without_admission(self) -> None:
        admitted_without_id = candidate_record(
            "lesson",
            "generated-probe",
            prompt="question: where is ada's orb?\nanswer:",
            target=" shelf.",
            state="admitted",
        )
        manifest = build_candidate_quarantine_manifest(
            "self-improvement-answer-cycle",
            "attempt-001",
            [admitted_without_id],
        )

        report = verify_candidate_quarantine_manifest(manifest)

        self.assertFalse(report["passed"])
        self.assertIn("training_eligible_candidates_have_admissions", report["failed_checks"])

    def test_training_plan_passes_clean_closed_world_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_candidate_quarantine_manifest(
                "transformer-answer-train",
                "run-001",
            )
            verifier_path = root / "run" / "closed_world_verifier.json"
            plan = build_training_plan(
                "transformer-answer-train",
                "run-001",
                root / "build" / "train.txt",
                root / "corpus",
                [root / "evals" / "qa.jsonl"],
                [SimpleNamespace(prompt="q1", target=" a.", source="qa:place")],
                [SimpleNamespace(prompt="q1", target=" a.", source="qa:place")],
                root / "run" / "corpus_hygiene.json",
                planned_artifacts=[verifier_path],
                candidate_quarantine_path=root / "run" / "candidate_quarantine.json",
                candidate_quarantine_summary=candidate_quarantine_summary(manifest),
            )
            hygiene = {
                "train_eval_overlap": {
                    "passed": True,
                    "protected_prompt_overlap_count": 0,
                    "protected_train_text_prompt_overlap_count": 0,
                }
            }

            report = verify_training_plan(
                plan,
                corpus_hygiene=hygiene,
                candidate_quarantine=manifest,
                subject_path=root / "run" / "training_plan.json",
                verifier_path=verifier_path,
            )
            updated = attach_verifier_summary(plan, report, verifier_path)
            write_verifier_report(verifier_path, report)
            written = json.loads(verifier_path.read_text(encoding="utf-8"))

        self.assertTrue(report["passed"])
        self.assertEqual(verifier_report_summary(report)["status"], "passed")
        self.assertEqual(updated["closed_world_verifier"]["status"], "written")
        self.assertTrue(written["passed"])

    def test_training_plan_rejects_pretrained_boundary_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_candidate_quarantine_manifest("transformer-answer-train", "run-001")
            verifier_path = root / "run" / "closed_world_verifier.json"
            plan = build_training_plan(
                "transformer-answer-train",
                "run-001",
                root / "build" / "train.txt",
                root / "corpus",
                [root / "evals" / "qa.jsonl"],
                [SimpleNamespace(prompt="q1", target=" a.", source="qa:place")],
                [SimpleNamespace(prompt="q1", target=" a.", source="qa:place")],
                root / "run" / "corpus_hygiene.json",
                planned_artifacts=[verifier_path],
                candidate_quarantine_path=root / "run" / "candidate_quarantine.json",
                candidate_quarantine_summary=candidate_quarantine_summary(manifest),
            )
            plan["data_boundary"]["pretrained_weights"] = True

            report = verify_training_plan(
                plan,
                corpus_hygiene={"train_eval_overlap": {"passed": True}},
                candidate_quarantine=manifest,
                verifier_path=verifier_path,
            )

        self.assertFalse(report["passed"])
        self.assertIn("closed_world_data_boundary", report["failed_checks"])

    def test_training_plan_rejects_candidate_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_candidate_quarantine_manifest("transformer-answer-train", "run-001")
            verifier_path = root / "run" / "closed_world_verifier.json"
            plan = build_training_plan(
                "transformer-answer-train",
                "run-001",
                root / "build" / "train.txt",
                root / "corpus",
                [root / "evals" / "qa.jsonl"],
                [SimpleNamespace(prompt="q1", target=" a.", source="candidate:repair")],
                [SimpleNamespace(prompt="q1", target=" a.", source="candidate:repair")],
                root / "run" / "corpus_hygiene.json",
                planned_artifacts=[verifier_path],
                candidate_quarantine_path=root / "run" / "candidate_quarantine.json",
                candidate_quarantine_summary=candidate_quarantine_summary(manifest),
            )

            report = verify_training_plan(
                plan,
                corpus_hygiene={"train_eval_overlap": {"passed": True}},
                candidate_quarantine=manifest,
                verifier_path=verifier_path,
            )

        self.assertFalse(report["passed"])
        self.assertIn("no_candidate_examples_in_training", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
