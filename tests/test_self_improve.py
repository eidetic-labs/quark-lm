from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.curriculum import build_curriculum, read_jsonl
from closed_world_lm.glossary_probes import DEFAULT_OUTPUT as DEFAULT_GLOSSARY_PROBES
from closed_world_lm.self_improve import (
    audit_exact_promotion,
    audit_forgetting,
    evaluate_responder,
    next_attempt,
    promotion_gate,
    self_improvement_experiment_intent,
    write_report_artifacts,
)


def current_admission_count() -> int:
    return len(read_jsonl(ROOT / "corpus" / "admissions.jsonl"))


class SelfImproveTest(unittest.TestCase):
    def test_responder_summary_tracks_all_eval_sets(self) -> None:
        curriculum = build_curriculum(seed=3)
        summary = evaluate_responder(curriculum.train_text)

        self.assertEqual(summary["qa"]["exact_rate"], 1.0)
        self.assertEqual(summary["unknowns"]["exact_rate"], 1.0)
        self.assertEqual(summary["heldout"]["exact_rate"], 1.0)
        self.assertEqual(summary["paraphrases"]["exact_rate"], 1.0)
        self.assertEqual(summary["owner"]["exact_rate"], 1.0)
        self.assertEqual(summary["self"]["exact_rate"], 1.0)
        self.assertEqual(summary["learning"]["exact_rate"], 1.0)
        self.assertEqual(summary["admissions"]["count"], current_admission_count() * 4)
        self.assertEqual(summary["admissions"]["exact_rate"], 1.0)
        self.assertEqual(summary["admission_paraphrases"]["count"], current_admission_count() * 7)
        self.assertEqual(summary["admission_paraphrases"]["exact_rate"], 1.0)
        self.assertEqual(summary["glossary"]["count"], len(read_jsonl(DEFAULT_GLOSSARY_PROBES)))
        self.assertEqual(summary["glossary"]["exact_rate"], 1.0)

    def test_cycle_paths_can_be_created_under_tempdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse((root / "self_improvement_report.json").exists())

    def test_next_attempt_advances_from_existing_attempt_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "attempts" / "attempt-001").mkdir(parents=True)
            (run_dir / "attempts" / "attempt-003").mkdir()
            (run_dir / "attempts" / "scratch").mkdir()

            number, path = next_attempt(run_dir)

            self.assertEqual(number, 4)
            self.assertEqual(path.name, "attempt-004")

    def test_self_improvement_experiment_intent_declares_required_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            attempt_dir = run_dir / "attempts" / "attempt-001"
            args = SimpleNamespace(
                corpus_dir=root / "corpus",
                experiment_version="v0.71",
                experiment_hypothesis=None,
                experiment_note=["test note"],
            )

            intent = self_improvement_experiment_intent(
                args,
                run_dir,
                attempt_dir,
                root / "build" / "train.txt",
            )

        gates = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn("promotion_gate", gates)
        self.assertIn("exact_eval_audit", gates)
        self.assertEqual(intent["decision"]["status"], "planned")
        self.assertEqual(intent["notes"], ["test note"])

    def test_write_report_artifacts_preserves_attempt_and_latest_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            attempt_dir = run_dir / "attempts" / "attempt-001"
            attempt_dir.mkdir(parents=True)
            args = SimpleNamespace(
                corpus_dir=run_dir / "corpus",
                experiment_version="v0.71",
                experiment_hypothesis=None,
                experiment_note=None,
            )
            report = {
                "corpus_snapshot": {"schema_version": 1},
                "corpus_diff": {"status": "evaluated"},
                "promotion_gate": {"passed": False},
                "experiment_intent": self_improvement_experiment_intent(
                    args,
                    run_dir,
                    attempt_dir,
                    run_dir / "build" / "train.txt",
                ),
            }

            write_report_artifacts(report, run_dir, attempt_dir, 1)

            attempt_report = json.loads(
                (attempt_dir / "self_improvement_report.json").read_text(encoding="utf-8")
            )
            latest_report = json.loads(
                (run_dir / "self_improvement_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(attempt_report["attempt"]["index"], 1)
            self.assertEqual(latest_report["attempt"]["report"], str(attempt_dir / "self_improvement_report.json"))
            self.assertTrue((attempt_dir / "corpus_snapshot.json").exists())
            self.assertTrue((run_dir / "corpus_diff.json").exists())
            self.assertTrue((attempt_dir / "experiment_intent.json").exists())
            self.assertTrue((run_dir / "experiment_intent.json").exists())

    def test_forgetting_audit_detects_regression(self) -> None:
        previous = {
            "responder": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}},
            "answer_model": {"final": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}}},
            "answer_decoder": {"final": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}}},
        }
        current = {
            "responder": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}},
            "answer_model": {"final": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}}},
            "answer_decoder": {"final": {"qa": {"count": 8, "exact": 7, "exact_rate": 0.875}}},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "previous.json"
            path.write_text(json.dumps(previous), encoding="utf-8")

            audit = audit_forgetting(current, path)

            self.assertFalse(audit["passed"])
            failed = [check for check in audit["checks"] if not check["passed"]]
            self.assertEqual(failed[0]["component"], "answer_decoder")

    def test_promotion_gate_detects_non_exact_eval(self) -> None:
        report = {
            "responder": {"qa": {"count": 1, "exact": 1}},
            "answer_model": {"final": {"qa": {"count": 1, "exact": 0}}},
            "answer_decoder": {"final": {"qa": {"count": 1, "exact": 1}}},
            "admission_probe_audit": {"passed": True},
            "glossary_probe_audit": {"passed": True},
            "prompt_leakage_audit": {
                "heldout": {"passed": True},
                "owner_heldout": {"passed": True},
            },
            "forgetting_audit": {"passed": True},
        }
        report["exact_eval_audit"] = audit_exact_promotion(report)

        gate = promotion_gate(report)

        self.assertFalse(report["exact_eval_audit"]["passed"])
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
