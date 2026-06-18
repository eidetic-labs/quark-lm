from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_improve import next_attempt, self_improvement_experiment_intent


class SelfImproveAttemptsTest(unittest.TestCase):
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
        self.assertIn("training_recipe", gates)
        self.assertIn("closed_world_verifier", gates)
        self.assertIn("tokenizer_candidate_guard", gates)
        self.assertIn("constraint_first_promotion", gates)
        self.assertIn("promotion_gate", gates)
        self.assertIn("exact_eval_audit", gates)
        self.assertIn(
            str(attempt_dir / "candidate_quarantine.json"),
            intent["planned_artifacts"],
        )
        self.assertIn(
            str(attempt_dir / "tokenizer_manifest.json"),
            intent["planned_artifacts"],
        )
        self.assertIn(
            str(attempt_dir / "tokenizer_report.json"),
            intent["planned_artifacts"],
        )
        self.assertIn(
            str(attempt_dir / "closed_world_verifier.json"),
            intent["planned_artifacts"],
        )
        self.assertIn(
            str(attempt_dir / "training_recipe.json"),
            intent["planned_artifacts"],
        )
        self.assertIn(
            str(attempt_dir / "constraint_first_promotion.json"),
            intent["planned_artifacts"],
        )
        self.assertEqual(intent["decision"]["status"], "planned")
        self.assertEqual(intent["notes"], ["test note"])


if __name__ == "__main__":
    unittest.main()
