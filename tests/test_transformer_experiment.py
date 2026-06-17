from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from support.experiment_modes import PROFILE_REPLAY_MODE_CASES, PROFILE_REPLAY_PLAN_PATH
from transformer_experiment import (
    TRAINING_DATA_DESCRIPTION,
    TRANSFORMER_RECIPE_VERSION,
    TransformerRunArtifacts,
    direct_answer_is_profile_aware,
    transformer_experiment_decision,
    transformer_experiment_intent,
    transformer_training_recipe,
    transformer_training_recipe_id,
)


class FakeTokenizer:
    vocab_size = 23


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        run=Path("runs/profile-screen"),
        train_text=Path("data/train.txt"),
        valid=Path("data/valid.txt"),
        corpus_dir=Path("corpus"),
        experiment_version=TRANSFORMER_RECIPE_VERSION,
        experiment_hypothesis=None,
        experiment_note=[],
        experiment_failure_criterion=[],
        experiment_acceptance_gate=[],
        resume_checkpoint=None,
        steps=3,
        learning_rate=0.01,
        eval_every=1,
        target_loss_weight=1.0,
        choice_loss_weight=0.0,
        choice_negatives=0,
        context_size=16,
        embedding_dim=8,
        feedforward_dim=16,
        direct_answer_steps=2,
        direct_answer_mode=(
            "branch-context-profile-coverage-preserving-deficit-unlikelihood"
        ),
        direct_answer_learning_rate=0.02,
        direct_answer_branch_position=1,
        direct_answer_branch_span=1,
        direct_answer_snapshot_mode="branch-only",
        direct_answer_require_branch_context_gate=True,
        seed=17,
    )


class TransformerExperimentTests(unittest.TestCase):
    def test_artifact_paths_keep_answer_training_contract_together(self) -> None:
        artifacts = TransformerRunArtifacts.from_run(
            Path("runs/profile-screen"),
            direct_profile_aware=True,
        )

        self.assertEqual(
            artifacts.training_recipe,
            Path("runs/profile-screen/training_recipe.json"),
        )
        self.assertEqual(
            artifacts.replay_plan,
            Path(PROFILE_REPLAY_PLAN_PATH),
        )
        self.assertEqual(
            artifacts.retrieval_memory,
            Path("runs/profile-screen/retrieval_memory_report.json"),
        )
        self.assertEqual(
            artifacts.memory_consolidation_plan,
            Path("runs/profile-screen/memory_consolidation_plan.json"),
        )
        self.assertIn(artifacts.constraint_first_promotion, artifacts.training_plan_artifacts())
        self.assertIn(artifacts.retrieval_memory, artifacts.training_plan_artifacts())
        self.assertIn(artifacts.memory_consolidation_plan, artifacts.training_plan_artifacts())
        self.assertIn(str(artifacts.replay_plan), artifacts.intent_artifacts())
        self.assertIn(str(artifacts.retrieval_memory), artifacts.intent_artifacts())
        self.assertIn(str(artifacts.memory_consolidation_plan), artifacts.intent_artifacts())

    def test_experiment_intent_uses_v078_recipe_and_artifact_surface(self) -> None:
        args = _args()
        intent = transformer_experiment_intent(args)

        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-coverage-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn("runs/profile-screen/training_recipe.json", intent["planned_artifacts"])
        self.assertIn(
            "runs/profile-screen/retrieval_memory_report.json",
            intent["planned_artifacts"],
        )
        self.assertIn(
            "runs/profile-screen/memory_consolidation_plan.json",
            intent["planned_artifacts"],
        )
        self.assertIn(PROFILE_REPLAY_PLAN_PATH, intent["planned_artifacts"])

    def test_profile_replay_modes_keep_profile_surface(self) -> None:
        for case in PROFILE_REPLAY_MODE_CASES:
            with self.subTest(mode=case.name):
                args = _args()
                args.direct_answer_mode = case.mode

                intent = transformer_experiment_intent(args)

                self.assertTrue(direct_answer_is_profile_aware(args))
                self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
                self.assertEqual(intent["training_recipe_id"], case.recipe_id)
                self.assertIn(PROFILE_REPLAY_PLAN_PATH, intent["planned_artifacts"])
                if case.expected_gate is not None:
                    gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
                    self.assertIn(case.expected_gate, gate_names)

    def test_training_recipe_records_external_boundaries(self) -> None:
        args = _args()
        artifacts = TransformerRunArtifacts.from_run(args.run, direct_profile_aware=True)
        recipe = transformer_training_recipe(
            args,
            FakeTokenizer(),
            artifacts.training_plan_artifacts(),
            transformer_experiment_intent(args)["acceptance_gates"],
            model_config={"vocab_size": 23, "context_size": 16},
            optimizer_config={"optimizer": "sgd"},
            generation_config={"temperature": 0.0},
            replay_plan_path=artifacts.replay_plan,
        )

        self.assertEqual(recipe["recipe_id"], transformer_training_recipe_id(args))
        self.assertFalse(recipe["uses_external_model"])
        self.assertEqual(recipe["data"]["training_examples"], TRAINING_DATA_DESCRIPTION)
        self.assertEqual(recipe["replay"]["status"], "planned")

    def test_decision_depends_on_constraint_first_gate(self) -> None:
        status, summary, evidence = transformer_experiment_decision(
            {
                "baseline": {"step": 0},
                "final": {"step": 1},
                "training_data": TRAINING_DATA_DESCRIPTION,
                "closed_world_verifier": {"passed": True},
                "training_recipe": {"recipe_id": "test"},
                "constraint_first_promotion": {
                    "passed": False,
                    "status": "blocked_before_quality_metrics",
                },
                "pretrained_weights": False,
                "pretrained_tokenizer": False,
                "external_embeddings": False,
            }
        )

        self.assertEqual(status, "rejected")
        self.assertIn("constraint-first", summary)
        by_name = {item["name"]: item for item in evidence}
        self.assertFalse(by_name["constraint_first_promotion"]["passed"])


if __name__ == "__main__":
    unittest.main()
