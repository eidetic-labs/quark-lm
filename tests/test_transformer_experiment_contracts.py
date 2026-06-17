from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from support.commands import (
    parse_args,
    train_transformer_answers,
    transformer_experiment_decision,
    transformer_experiment_intent,
    transformer_training_recipe,
    transformer_training_recipe_id,
)
from support.core import CharTokenizer


class TransformerExperimentContractsTest(unittest.TestCase):
    def test_transformer_experiment_intent_records_profile_aware_plan(self) -> None:
        args = SimpleNamespace(
            train_text=Path("build/train.txt"),
            valid=Path("build/valid.txt"),
            corpus_dir=Path("corpus"),
            run=Path("runs/profile-screen"),
            direct_answer_steps=1,
            direct_answer_mode=(
                "branch-context-profile-coverage-preserving-deficit-unlikelihood"
            ),
            experiment_version="v0.71",
            experiment_hypothesis="Profile-aware screens should declare their replay plan.",
            experiment_acceptance_gate=["custom_gate:Custom rule."],
            experiment_failure_criterion=["Custom failure."],
            experiment_note=["Custom note."],
        )

        intent = transformer_experiment_intent(args)

        gates = {gate["name"] for gate in intent["acceptance_gates"]}
        self.assertIn("training_recipe", gates)
        self.assertIn("closed_world_verifier", gates)
        self.assertIn("constraint_first_promotion", gates)
        self.assertIn("branch_context_gate_recorded", gates)
        self.assertIn("custom_gate", gates)
        self.assertEqual(
            intent["training_recipe_id"],
            "transformer-answer:branch-context-profile-coverage-preserving-deficit-unlikelihood:v0.78",
        )
        self.assertEqual(intent["replay_plan_id"], "direct_answer_replay_plan.json")
        self.assertIn(
            "runs/profile-screen/candidate_quarantine.json",
            intent["planned_artifacts"],
        )
        self.assertIn(
            "runs/profile-screen/closed_world_verifier.json",
            intent["planned_artifacts"],
        )
        self.assertIn(
            "runs/profile-screen/training_recipe.json",
            intent["planned_artifacts"],
        )
        self.assertIn(
            "runs/profile-screen/constraint_first_promotion.json",
            intent["planned_artifacts"],
        )
        self.assertEqual(intent["decision"]["status"], "planned")

    def test_transformer_training_recipe_records_replay_and_rerun_surface(self) -> None:
        args = _recipe_args()
        tokenizer = CharTokenizer.train("abc")

        recipe = transformer_training_recipe(
            args,
            tokenizer,
            [Path("runs/profile-screen/training_recipe.json")],
            [{"name": "gate", "rule": "Gate.", "required": True}],
            Path("runs/profile-screen/direct_answer_replay_plan.json"),
        )

        self.assertEqual(recipe["recipe_id"], transformer_training_recipe_id(args))
        self.assertEqual(recipe["tokenizer"]["vocab_size"], tokenizer.vocab_size)
        self.assertEqual(recipe["optimizer"]["optimizer"], "adamw")
        self.assertEqual(recipe["replay"]["status"], "planned")
        self.assertEqual(
            recipe["rerun"]["entry_point"],
            "quark-lm-transformer answer-train",
        )

    def test_transformer_experiment_decision_records_screen_evidence(self) -> None:
        metrics = {
            "baseline": {"step": 0},
            "final": {"step": 1},
            "training_data": (
                "answer_model corpus-derived AnswerExample lessons"
            ),
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "closed_world_verifier": {"passed": True},
            "training_recipe": {"recipe_id": "transformer-answer:test:v0.78"},
            "constraint_first_promotion": {
                "passed": False,
                "status": "blocked_before_quality_metrics",
            },
            "direct_answer": {
                "direct_answer_branch_context_gate": {"passed": True},
                "final": {
                    "branch_diversity_target": {"passed": False},
                    "branch_target_coverage_by_profile": {"qa": {"a": 1}},
                },
            },
        }

        status, summary, evidence = transformer_experiment_decision(metrics)

        evidence_by_name = {item["name"]: item for item in evidence}
        self.assertEqual(status, "rejected")
        self.assertIn("constraint-first promotion gate", summary)
        self.assertFalse(evidence_by_name["constraint_first_promotion"]["passed"])
        self.assertTrue(evidence_by_name["branch_context_gate_recorded"]["passed"])
        self.assertFalse(evidence_by_name["branch_diversity_target"]["passed"])

    def test_transformer_answer_metrics_declare_external_embedding_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "answer-screen"),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--direct-answer-steps",
                    "0",
                    "--selector-steps",
                    "0",
                    "--generator-steps",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "8",
                ]
            )

            metrics = train_transformer_answers(args)

        self.assertFalse(metrics["pretrained_weights"])
        self.assertFalse(metrics["pretrained_tokenizer"])
        self.assertFalse(metrics["external_embeddings"])
        failed_constraints = metrics["constraint_first_promotion"]["failed_constraints"]
        self.assertNotIn("no_external_embeddings", failed_constraints)


def _recipe_args() -> SimpleNamespace:
    return SimpleNamespace(
        train_text=Path("build/train.txt"),
        valid=Path("build/valid.txt"),
        corpus_dir=Path("corpus"),
        run=Path("runs/profile-screen"),
        context_size=16,
        embedding_dim=4,
        feedforward_dim=8,
        num_layers=1,
        attention_heads=1,
        seed=17,
        use_layer_norm=False,
        use_pre_layer_norm=False,
        use_rms_norm=True,
        layer_norm_epsilon=1e-5,
        use_gated_mlp=True,
        tie_output_embeddings=True,
        use_rotary_positions=True,
        use_kv_cache_path=False,
        use_context_mean=False,
        use_context_projection=False,
        use_prompt_prefix_projection=False,
        use_prompt_position_projection=False,
        prompt_position_projection_scale=1.0,
        use_prompt_attention_summary=False,
        resume_checkpoint=None,
        steps=5,
        learning_rate=0.03,
        eval_every=5,
        target_loss_weight=1.0,
        choice_loss_weight=0.0,
        choice_negatives=0,
        direct_answer_steps=1,
        direct_answer_mode=(
            "branch-context-profile-coverage-preserving-deficit-unlikelihood"
        ),
        direct_answer_learning_rate=0.01,
        direct_answer_branch_position=1,
        direct_answer_branch_span=1,
        direct_answer_snapshot_mode="branch-only",
        direct_answer_require_branch_context_gate=True,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        trace_top_tokens=5,
        use_kv_cache=False,
        optimizer="adamw",
        gradient_clip=5.0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        warmup_steps=1,
        decay_steps=10,
        min_learning_rate=0.0,
        gradient_accumulation_steps=2,
        experiment_version="v0.78",
    )


if __name__ == "__main__":
    unittest.main()
