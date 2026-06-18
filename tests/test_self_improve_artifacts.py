from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate_quarantine import build_candidate_quarantine_manifest
from closed_world_verifier_reports import verifier_check, verifier_report
from constraint_first_report import (
    build_constraint_first_promotion_report,
    promotion_check,
)
from self_improve import self_improvement_experiment_intent, write_report_artifacts
from self_improvement_tokenizer import tokenizer_candidate_record
from training_recipe_core import build_training_recipe


class SelfImproveArtifactsTest(unittest.TestCase):
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
                "corpus_hygiene": {
                    "schema_version": 1,
                    "kind": "corpus_hygiene_report",
                },
                "training_plan": {"schema_version": 1, "kind": "training_plan"},
                "training_recipe": build_training_recipe(
                    version="v0.77",
                    component="self-improvement-answer-cycle",
                    run_id="attempt-001",
                    recipe_id="self-improve-answer-cycle:v0.77",
                    purpose="Test recipe.",
                    model={"component": "answer"},
                    tokenizer={"type": "char"},
                    data={"train_text": "build/train.txt"},
                    objective={"mode": "answer-cycle"},
                    optimizer={"seed": 7},
                    artifacts=["training_recipe.json"],
                    gates=[
                        {
                            "name": "training_recipe",
                            "rule": "Recipe exists.",
                            "required": True,
                        }
                    ],
                ),
                "candidate_quarantine": build_candidate_quarantine_manifest(
                    "self-improvement-answer-cycle",
                    "attempt-001",
                ),
                "tokenizer_candidate": tokenizer_candidate_record(
                    attempt_dir / "tokenizer_manifest.json",
                    attempt_dir / "tokenizer_report.json",
                    "abc123",
                    {
                        "tokenizer_type": "closed-world-subword",
                        "corpus_hash": "hash",
                        "purity": {
                            "pretrained_tokenizer": False,
                            "external_vocabulary": False,
                            "admitted_corpus_only": True,
                        },
                        "rejected_candidates": [],
                    },
                    {
                        "round_trip_ok": True,
                        "accepted_token_count": 1,
                        "token_count_savings": 2,
                        "compression_ratio": 0.9,
                        "branch_diversity_score": 1.0,
                        "full_answer_tokens": [],
                    },
                ),
                "closed_world_verifier": verifier_report(
                    "self-improvement-answer-cycle",
                    "attempt-001",
                    "training_plan",
                    [
                        verifier_check(
                            "test_verifier",
                            True,
                            "Test verifier evidence passes.",
                        )
                    ],
                ),
                "constraint_first_promotion": build_constraint_first_promotion_report(
                    "self-improvement-answer-cycle",
                    "attempt-001",
                    "self_improvement_report",
                    [promotion_check("constraint", True, "Constraint passes.")],
                    [promotion_check("quality", True, "Quality passes.")],
                ),
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
                (attempt_dir / "self_improvement_report.json").read_text(
                    encoding="utf-8"
                )
            )
            latest_report = json.loads(
                (run_dir / "self_improvement_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(attempt_report["attempt"]["index"], 1)
            self.assertEqual(
                latest_report["attempt"]["report"],
                str(attempt_dir / "self_improvement_report.json"),
            )
            self.assertTrue((attempt_dir / "corpus_snapshot.json").exists())
            self.assertTrue((run_dir / "corpus_diff.json").exists())
            self.assertTrue((attempt_dir / "corpus_hygiene.json").exists())
            self.assertTrue((run_dir / "training_plan.json").exists())
            self.assertTrue((attempt_dir / "training_recipe.json").exists())
            self.assertTrue((run_dir / "training_recipe.json").exists())
            self.assertTrue((attempt_dir / "candidate_quarantine.json").exists())
            self.assertTrue((run_dir / "candidate_quarantine.json").exists())
            self.assertTrue((attempt_dir / "tokenizer_manifest.json").exists())
            self.assertTrue((run_dir / "tokenizer_manifest.json").exists())
            self.assertTrue((attempt_dir / "tokenizer_report.json").exists())
            self.assertTrue((run_dir / "tokenizer_report.json").exists())
            self.assertTrue((attempt_dir / "closed_world_verifier.json").exists())
            self.assertTrue((run_dir / "closed_world_verifier.json").exists())
            self.assertTrue((attempt_dir / "constraint_first_promotion.json").exists())
            self.assertTrue((run_dir / "constraint_first_promotion.json").exists())
            self.assertTrue((attempt_dir / "experiment_intent.json").exists())
            self.assertTrue((run_dir / "experiment_intent.json").exists())


if __name__ == "__main__":
    unittest.main()
