from __future__ import annotations

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.transformer_experiment import (
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
            Path("runs/profile-screen/direct_answer_replay_plan.json"),
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
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_target_share_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_prompt_ownership_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_baseline_anchored_prompt_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_baseline_floor_gated_prompt_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_baseline_floor_adaptive_prompt_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_baseline_floor_repaired_prompt_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_baseline_floor_objective_prompt_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_baseline_floor_stabilization_mode_keeps_profile_replay_surface(self) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-stabilization-unlikelihood:"
                "v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_targeted_stabilization_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-profile-targeted-"
            "stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-profile-targeted-"
                "stabilization-unlikelihood:v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_sequential_stabilization_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-sequential-profile-"
            "stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-sequential-profile-"
                "stabilization-unlikelihood:v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_calibrated_sequential_stabilization_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-calibrated-sequential-"
            "profile-stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-calibrated-sequential-"
                "profile-stabilization-unlikelihood:v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_calibrated_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-profile-scale-calibrated-"
            "sequential-profile-stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-profile-scale-calibrated-"
                "sequential-profile-stabilization-unlikelihood:v0.78"
            ),
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_diversity_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-profile-scale-"
            "calibrated-sequential-profile-stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-profile-scale-"
                "calibrated-sequential-profile-stabilization-unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_diversity_calibrated_"
                "sequential_stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_frontier_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-frontier-profile-"
            "scale-calibrated-sequential-profile-stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-frontier-profile-"
                "scale-calibrated-sequential-profile-stabilization-unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_frontier_calibrated_"
                "sequential_stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_coverage_frontier_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-coverage-frontier-"
            "profile-scale-calibrated-sequential-profile-stabilization-"
            "unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-coverage-"
                "frontier-profile-scale-calibrated-sequential-profile-"
                "stabilization-unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_coverage_frontier_"
                "calibrated_sequential_stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_coverage_prep_frontier_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-coverage-prep-"
            "frontier-profile-scale-calibrated-sequential-profile-"
            "stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-coverage-prep-"
                "frontier-profile-scale-calibrated-sequential-profile-"
                "stabilization-unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_coverage_prep_frontier_"
                "calibrated_sequential_stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_coverage_recovery_frontier_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-coverage-recovery-"
            "frontier-profile-scale-calibrated-sequential-profile-"
            "stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-coverage-recovery-"
                "frontier-profile-scale-calibrated-sequential-profile-"
                "stabilization-unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_coverage_recovery_frontier_"
                "calibrated_sequential_stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_branch_stable_coverage_recovery_frontier_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-branch-stable-"
            "coverage-recovery-frontier-profile-scale-calibrated-sequential-"
            "profile-stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-branch-stable-"
                "coverage-recovery-frontier-profile-scale-calibrated-sequential-"
                "profile-stabilization-unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_branch_stable_"
                "coverage_recovery_frontier_calibrated_sequential_"
                "stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_branch_diversity_recovery_frontier_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-branch-stable-"
            "coverage-recovery-branch-diversity-frontier-profile-scale-"
            "calibrated-sequential-profile-stabilization-unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-branch-stable-"
                "coverage-recovery-branch-diversity-frontier-profile-scale-"
                "calibrated-sequential-profile-stabilization-unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_branch_diversity_"
                "recovery_frontier_calibrated_sequential_"
                "stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

    def test_profile_scale_memory_consolidation_frontier_mode_keeps_profile_replay_surface(
        self,
    ) -> None:
        args = _args()
        args.direct_answer_mode = (
            "branch-context-profile-baseline-floor-diversity-branch-stable-"
            "coverage-recovery-branch-diversity-collapsed-profile-binding-"
            "remaining-profile-owner-paraphrase-memory-consolidation-frontier-"
            "profile-scale-calibrated-sequential-profile-stabilization-"
            "unlikelihood"
        )

        intent = transformer_experiment_intent(args)

        self.assertTrue(direct_answer_is_profile_aware(args))
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertEqual(
            intent["training_recipe_id"],
            (
                "transformer-answer:"
                "branch-context-profile-baseline-floor-diversity-branch-stable-"
                "coverage-recovery-branch-diversity-collapsed-profile-binding-"
                "remaining-profile-owner-paraphrase-memory-consolidation-frontier-"
                "profile-scale-calibrated-sequential-profile-stabilization-"
                "unlikelihood:v0.78"
            ),
        )
        gate_names = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn(
            (
                "baseline_floor_profile_scale_memory_consolidation_"
                "frontier_calibrated_sequential_stabilization_screen"
            ),
            gate_names,
        )
        self.assertIn(
            "runs/profile-screen/direct_answer_replay_plan.json",
            intent["planned_artifacts"],
        )

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
