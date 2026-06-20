from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constraint_first_report import (
    build_constraint_first_promotion_report,
    constraint_first_summary,
    promotion_check,
    write_constraint_first_report,
)
from self_improvement_constraints import self_improvement_constraint_report
from training_recipe_core import (
    attach_recipe_summary,
    build_training_recipe,
    write_training_recipe,
)
from transformer_backend_policy import transformer_backend_metadata
from transformer_constraints import transformer_constraint_report


def minimal_recipe() -> dict:
    return build_training_recipe(
        version="v0.77",
        component="component",
        run_id="run-001",
        recipe_id="component:test:v0.77",
        purpose="Test recipe.",
        model={"architecture": "tiny"},
        tokenizer={"type": "char", "pretrained_tokenizer": False},
        data={"train_text": "build/train.txt"},
        objective={"mode": "test"},
        optimizer={"optimizer": "sgd"},
        artifacts=["training_recipe.json"],
        gates=[{"name": "gate", "rule": "Gate must pass.", "required": True}],
    )


class TrainingRecipeTest(unittest.TestCase):
    def test_training_recipe_writes_and_attaches_summary(self) -> None:
        recipe = minimal_recipe()
        plan = {"schema_version": 1, "kind": "training_plan"}

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "training_recipe.json"
            write_training_recipe(path, recipe)
            written = json.loads(path.read_text(encoding="utf-8"))
            updated = attach_recipe_summary(plan, written, path)

        self.assertEqual(written["kind"], "training_recipe")
        self.assertFalse(written["uses_external_model"])
        self.assertEqual(updated["training_recipe"]["summary"]["recipe_id"], recipe["recipe_id"])

    def test_constraint_first_blocks_quality_when_constraints_fail(self) -> None:
        report = build_constraint_first_promotion_report(
            "transformer-answer-train",
            "run-001",
            "transformer_answer_metrics",
            constraints=[
                promotion_check("closed_world_verifier", False, "Verifier must pass."),
            ],
            quality_checks=[
                promotion_check("rank_improved", True, "Rank improved."),
            ],
        )

        self.assertEqual(report["status"], "blocked_before_quality_metrics")
        self.assertFalse(report["quality_metrics_considered"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_quality_checks"], [])

    def test_constraint_first_can_pass_after_constraints_and_quality_pass(self) -> None:
        report = build_constraint_first_promotion_report(
            "self-improvement-answer-cycle",
            "attempt-001",
            "self_improvement_report",
            constraints=[promotion_check("verifier", True, "Verifier passes.")],
            quality_checks=[promotion_check("exact_eval", True, "Exact eval passes.")],
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "constraint_first_promotion.json"
            write_constraint_first_report(path, report)
            written = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(written["status"], "eligible_for_promotion")
        self.assertTrue(constraint_first_summary(written)["passed"])
        self.assertTrue(written["quality_metrics_considered"])

    def test_self_improvement_constraint_report_uses_exact_eval_as_quality(self) -> None:
        report = self_improvement_constraint_report(
            {
                "run_id": "attempt-001",
                "closed_world_verifier": {"passed": True},
                "admission_probe_audit": {"passed": True},
                "glossary_probe_audit": {"passed": True},
                "tokenizer_candidate_guard": {"passed": True},
                "prompt_leakage_audit": {
                    "heldout": {"passed": True},
                    "owner_heldout": {"passed": True},
                },
                "forgetting_audit": {"passed": True},
                "exact_eval_audit": {"passed": True},
            }
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "eligible_for_promotion")
        self.assertTrue(report["quality_metrics_considered"])

    def test_self_improvement_constraint_report_blocks_missing_tokenizer_guard(self) -> None:
        report = self_improvement_constraint_report(
            {
                "run_id": "attempt-001",
                "closed_world_verifier": {"passed": True},
                "admission_probe_audit": {"passed": True},
                "glossary_probe_audit": {"passed": True},
                "prompt_leakage_audit": {
                    "heldout": {"passed": True},
                    "owner_heldout": {"passed": True},
                },
                "forgetting_audit": {"passed": True},
                "exact_eval_audit": {"passed": True},
            }
        )

        self.assertFalse(report["passed"])
        self.assertIn("tokenizer_candidate_guard", report["failed_constraints"])

    def test_transformer_constraint_report_blocks_on_diversity_before_quality(self) -> None:
        metrics = {
            "run_id": "run-001",
            "baseline": {"step": 0},
            "final": {"step": 1},
            "training_data": "answer_model corpus-derived AnswerExample lessons",
            "closed_world_verifier": {"passed": True},
            **_transformer_control_reports(),
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "backend": transformer_backend_metadata(
                seed=17,
                tokenizer_type="char",
            ),
            "direct_answer": {
                "direct_answer_weight_update_outcome": {
                    "status": "accepted",
                    "accepted": True,
                },
                "direct_answer_branch_context_gate": {"passed": True},
                "baseline": {"branch_target_coverage_by_profile": {"qa": 0.5}},
                "final": {
                    "branch_diversity_target": {"passed": False},
                    "branch_target_coverage_by_profile": {"qa": 0.5},
                    "evals": {"qa": {"count": 1, "exact": 1}},
                },
            },
        }

        report = transformer_constraint_report(metrics)

        self.assertEqual(report["status"], "blocked_before_quality_metrics")
        self.assertIn("branch_diversity_target", report["failed_constraints"])
        self.assertFalse(report["quality_metrics_considered"])

    def test_transformer_constraint_report_considers_exact_quality_after_constraints(self) -> None:
        metrics = {
            "run_id": "run-001",
            "baseline": {"step": 0},
            "final": {"step": 1},
            "training_data": "answer_model corpus-derived AnswerExample lessons",
            "closed_world_verifier": {"passed": True},
            **_transformer_control_reports(),
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "backend": transformer_backend_metadata(
                seed=17,
                tokenizer_type="char",
            ),
            "direct_answer": {
                "direct_answer_weight_update_outcome": {
                    "status": "accepted",
                    "accepted": True,
                },
                "direct_answer_branch_context_gate": {"passed": True},
                "baseline": {"branch_target_coverage_by_profile": {"qa": 1.0}},
                "final": {
                    "branch_diversity_target": {"passed": True},
                    "branch_target_coverage_by_profile": {"qa": 1.0},
                    "evals": {"qa": {"count": 1, "exact": 0}},
                },
            },
        }

        report = transformer_constraint_report(metrics)

        self.assertEqual(report["status"], "blocked_by_quality_checks")
        self.assertTrue(report["constraints_passed"])
        self.assertTrue(report["quality_metrics_considered"])
        self.assertEqual(report["failed_quality_checks"], ["direct_greedy_exact"])


def _transformer_control_reports() -> dict:
    return {
        "sweep_plan": {"kind": "transformer_sweep_plan"},
        "sweep_plan_path": "runs/run-001/sweep_plan.json",
        "replay_mixture_report": {"summary": {"passed": True}},
        "replay_mixture_report_path": "runs/run-001/replay_mixture_report.json",
    }


if __name__ == "__main__":
    unittest.main()
