from __future__ import annotations

import random
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
from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    TransformerGuidedAnswerGenerator,
    build_answer_selector,
    build_transformer_answer_generator,
    direct_answer_lesson,
    evaluate_answer_generator_records,
    evaluate_answer_records,
    evaluate_direct_answer_records,
    train_direct_answer_first_error,
)


class TransformerCliEvalTest(unittest.TestCase):
    def test_direct_answer_modes_include_rollout_and_hybrid(self) -> None:
        for mode in (
            "rollout-unlikelihood",
            "hybrid-unlikelihood",
            "staged-unlikelihood",
            "periodic-rollout-unlikelihood",
            "early-stop-unlikelihood",
            "periodic-early-stop-unlikelihood",
            "repeat-loop-unlikelihood",
            "periodic-repeat-loop-unlikelihood",
            "balanced-repair-unlikelihood",
            "periodic-balanced-repair-unlikelihood",
            "generated-prefix-recovery-unlikelihood",
            "periodic-generated-prefix-recovery-unlikelihood",
            "sequence-repair-unlikelihood",
            "periodic-sequence-repair-unlikelihood",
            "loop-escape-unlikelihood",
            "periodic-loop-escape-unlikelihood",
            "periodic-sequence-loop-escape-unlikelihood",
            "branch-repair-unlikelihood",
            "periodic-branch-repair-unlikelihood",
            "branch-collapse-unlikelihood",
            "periodic-branch-collapse-unlikelihood",
            "branch-batch-contrast-unlikelihood",
            "periodic-branch-batch-contrast-unlikelihood",
            "branch-diversity-unlikelihood",
            "periodic-branch-diversity-unlikelihood",
            "branch-target-softmax-unlikelihood",
            "periodic-branch-target-softmax-unlikelihood",
            "branch-target-margin-unlikelihood",
            "periodic-branch-target-margin-unlikelihood",
            "branch-hidden-projection-margin-unlikelihood",
            "branch-representation-contrast-unlikelihood",
            "branch-balanced-representation-contrast-unlikelihood",
            "branch-output-binding-unlikelihood",
            "branch-bidirectional-binding-unlikelihood",
            "branch-balanced-bidirectional-binding-unlikelihood",
            "branch-coverage-binding-unlikelihood",
            "branch-balanced-coverage-binding-unlikelihood",
            "branch-target-set-coverage-unlikelihood",
            "branch-balanced-target-set-coverage-unlikelihood",
            "branch-target-diversity-unlikelihood",
            "branch-balanced-target-diversity-unlikelihood",
            "branch-target-replay-coverage-unlikelihood",
            "branch-balanced-target-replay-coverage-unlikelihood",
            "branch-context-replay-coverage-unlikelihood",
            "branch-balanced-context-replay-coverage-unlikelihood",
            "branch-context-coverage-anchor-unlikelihood",
            "branch-balanced-context-coverage-anchor-unlikelihood",
            "branch-context-target-balanced-anchor-unlikelihood",
            "branch-balanced-context-target-balanced-anchor-unlikelihood",
            "branch-context-coverage-deficit-unlikelihood",
            "branch-balanced-context-coverage-deficit-unlikelihood",
            "branch-context-coverage-preserving-deficit-unlikelihood",
            "branch-balanced-context-coverage-preserving-deficit-unlikelihood",
            "branch-context-profile-coverage-preserving-deficit-unlikelihood",
            "branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood",
            "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood",
            "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            "branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            (
                "branch-balanced-context-profile-baseline-floor-gated-"
                "prompt-ownership-target-share-preserving-deficit-unlikelihood"
            ),
            (
                "branch-balanced-context-profile-baseline-floor-adaptive-"
                "prompt-ownership-target-share-preserving-deficit-unlikelihood"
            ),
            (
                "branch-balanced-context-profile-baseline-floor-repaired-"
                "prompt-ownership-target-share-preserving-deficit-unlikelihood"
            ),
            (
                "branch-balanced-context-profile-baseline-floor-objective-"
                "prompt-ownership-target-share-preserving-deficit-unlikelihood"
            ),
            "branch-context-profile-baseline-floor-stabilization-unlikelihood",
            (
                "branch-context-profile-baseline-floor-profile-targeted-"
                "stabilization-unlikelihood"
            ),
            (
                "branch-context-profile-baseline-floor-sequential-profile-"
                "stabilization-unlikelihood"
            ),
            (
                "branch-context-profile-baseline-floor-calibrated-sequential-"
                "profile-stabilization-unlikelihood"
            ),
            (
                "branch-context-profile-baseline-floor-profile-scale-calibrated-"
                "sequential-profile-stabilization-unlikelihood"
            ),
            "branch-rank-margin-unlikelihood",
            "branch-balanced-rank-margin-unlikelihood",
            "branch-topk-softmax-unlikelihood",
            "branch-balanced-topk-softmax-unlikelihood",
            "periodic-branch-representation-contrast-unlikelihood",
            "branch-span-repair-unlikelihood",
            "periodic-branch-span-repair-unlikelihood",
            "branch-contrast-unlikelihood",
            "periodic-branch-contrast-unlikelihood",
            "branch-span-contrast-unlikelihood",
            "periodic-branch-span-contrast-unlikelihood",
            "hard-branch-contrast-unlikelihood",
            "periodic-hard-branch-contrast-unlikelihood",
            "periodic-branch-repair-contrast-unlikelihood",
            "periodic-branch-span-repair-contrast-unlikelihood",
            "periodic-hard-branch-repair-contrast-unlikelihood",
        ):
            args = parse_args(
                [
                    "answer-train",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-mode",
                    mode,
                    "--direct-answer-rollout-interval",
                    "4",
                    "--direct-answer-positive-weight",
                    "1.5",
                    "--direct-answer-contrast-weight",
                    "1.25",
                    "--direct-answer-recovery-steps",
                    "2",
                    "--direct-answer-branch-position",
                    "1",
                    "--direct-answer-branch-span",
                    "3",
                    "--direct-answer-branch-batch-size",
                    "5",
                    "--direct-answer-hard-negatives",
                    "7",
                    "--temperature",
                    "0.8",
                    "--top-k",
                    "3",
                    "--top-p",
                    "0.75",
                    "--repetition-penalty",
                    "1.2",
                    "--trace-top-tokens",
                    "4",
                    "--use-kv-cache",
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-train-top-layer-only",
                    "--direct-answer-freeze-output-bias",
                    "--direct-answer-restore-best-branch-snapshot",
                    "--direct-answer-require-branch-context-gate",
                    "--skip-post-direct-snapshot",
                    "--direct-answer-sequence-interval",
                    "6",
                    "--num-layers",
                    "2",
                    "--attention-heads",
                    "2",
                    "--use-layer-norm",
                    "--use-pre-layer-norm",
                    "--use-rms-norm",
                    "--layer-norm-epsilon",
                    "0.0001",
                    "--use-gated-mlp",
                    "--tie-output-embeddings",
                    "--use-rotary-positions",
                    "--use-kv-cache-path",
                    "--use-context-mean",
                    "--use-context-projection",
                    "--use-prompt-prefix-projection",
                    "--use-prompt-position-projection",
                    "--prompt-position-projection-scale",
                    "12.5",
                    "--use-prompt-attention-summary",
                    "--optimizer",
                    "adamw",
                    "--gradient-clip",
                    "3.5",
                    "--weight-decay",
                    "0.01",
                    "--adam-beta1",
                    "0.8",
                    "--adam-beta2",
                    "0.95",
                    "--adam-epsilon",
                    "0.0000001",
                    "--warmup-steps",
                    "2",
                    "--decay-steps",
                    "7",
                    "--min-learning-rate",
                    "0.001",
                    "--gradient-accumulation-steps",
                    "2",
                ]
            )
            self.assertEqual(args.direct_answer_mode, mode)
            self.assertEqual(args.direct_answer_rollout_interval, 4)
            self.assertEqual(args.direct_answer_positive_weight, 1.5)
            self.assertEqual(args.direct_answer_contrast_weight, 1.25)
            self.assertEqual(args.direct_answer_recovery_steps, 2)
            self.assertEqual(args.direct_answer_branch_position, 1)
            self.assertEqual(args.direct_answer_branch_span, 3)
            self.assertEqual(args.direct_answer_branch_batch_size, 5)
            self.assertEqual(args.direct_answer_hard_negatives, 7)
            self.assertEqual(args.temperature, 0.8)
            self.assertEqual(args.top_k, 3)
            self.assertEqual(args.top_p, 0.75)
            self.assertEqual(args.repetition_penalty, 1.2)
            self.assertEqual(args.trace_top_tokens, 4)
            self.assertTrue(args.use_kv_cache)
            self.assertEqual(args.direct_answer_snapshot_mode, "branch-only")
            self.assertTrue(args.direct_answer_train_top_layer_only)
            self.assertTrue(args.direct_answer_freeze_output_bias)
            self.assertTrue(args.direct_answer_restore_best_branch_snapshot)
            self.assertTrue(args.direct_answer_require_branch_context_gate)
            self.assertTrue(args.skip_post_direct_snapshot)
            self.assertEqual(args.num_layers, 2)
            self.assertEqual(args.attention_heads, 2)
            self.assertTrue(args.use_layer_norm)
            self.assertTrue(args.use_pre_layer_norm)
            self.assertTrue(args.use_rms_norm)
            self.assertEqual(args.layer_norm_epsilon, 0.0001)
            self.assertTrue(args.use_gated_mlp)
            self.assertTrue(args.tie_output_embeddings)
            self.assertTrue(args.use_rotary_positions)
            self.assertTrue(args.use_kv_cache_path)
            self.assertTrue(args.use_context_mean)
            self.assertTrue(args.use_context_projection)
            self.assertTrue(args.use_prompt_prefix_projection)
            self.assertTrue(args.use_prompt_position_projection)
            self.assertEqual(args.prompt_position_projection_scale, 12.5)
            self.assertTrue(args.use_prompt_attention_summary)
            self.assertEqual(args.direct_answer_sequence_interval, 6)
            self.assertEqual(args.optimizer, "adamw")
            self.assertEqual(args.gradient_clip, 3.5)
            self.assertEqual(args.weight_decay, 0.01)
            self.assertEqual(args.adam_beta1, 0.8)
            self.assertEqual(args.adam_beta2, 0.95)
            self.assertEqual(args.adam_epsilon, 0.0000001)
            self.assertEqual(args.warmup_steps, 2)
            self.assertEqual(args.decay_steps, 7)
            self.assertEqual(args.min_learning_rate, 0.001)
            self.assertEqual(args.gradient_accumulation_steps, 2)

    def test_parse_train_args_accepts_context_mean(self) -> None:
        args = parse_args(
            [
                "train",
                "--use-context-mean",
                "--use-context-projection",
                "--use-prompt-prefix-projection",
                "--use-prompt-position-projection",
                "--use-pre-layer-norm",
                "--use-rms-norm",
                "--prompt-position-projection-scale",
                "4.0",
                "--use-prompt-attention-summary",
                "--attention-heads",
                "2",
                "--use-gated-mlp",
                "--tie-output-embeddings",
                "--use-rotary-positions",
                "--use-kv-cache-path",
                "--optimizer",
                "adamw",
                "--gradient-accumulation-steps",
                "3",
            ]
        )

        self.assertTrue(args.use_context_mean)
        self.assertTrue(args.use_context_projection)
        self.assertTrue(args.use_prompt_prefix_projection)
        self.assertTrue(args.use_prompt_position_projection)
        self.assertTrue(args.use_pre_layer_norm)
        self.assertTrue(args.use_rms_norm)
        self.assertEqual(args.prompt_position_projection_scale, 4.0)
        self.assertTrue(args.use_prompt_attention_summary)
        self.assertEqual(args.attention_heads, 2)
        self.assertTrue(args.use_gated_mlp)
        self.assertTrue(args.tie_output_embeddings)
        self.assertTrue(args.use_rotary_positions)
        self.assertTrue(args.use_kv_cache_path)
        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.gradient_accumulation_steps, 3)

    def test_parse_answer_args_accepts_experiment_contract(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--experiment-version",
                "v0.71",
                "--experiment-hypothesis",
                "A bounded screen can explain itself.",
                "--experiment-acceptance-gate",
                "custom_gate:Custom rule.",
                "--experiment-failure-criterion",
                "Custom failure.",
                "--experiment-note",
                "Custom note.",
            ]
        )

        self.assertEqual(args.experiment_version, "v0.71")
        self.assertEqual(args.experiment_hypothesis, "A bounded screen can explain itself.")
        self.assertEqual(args.experiment_acceptance_gate, ["custom_gate:Custom rule."])
        self.assertEqual(args.experiment_failure_criterion, ["Custom failure."])
        self.assertEqual(args.experiment_note, ["Custom note."])

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
        args = SimpleNamespace(
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

    def test_parse_eval_args_accepts_generation_trace_controls(self) -> None:
        args = parse_args(
            [
                "eval",
                "--temperature",
                "0.6",
                "--top-k",
                "4",
                "--top-p",
                "0.8",
                "--repetition-penalty",
                "1.3",
                "--trace-top-tokens",
                "3",
                "--use-kv-cache",
                "--samples-jsonl",
                "samples.jsonl",
            ]
        )

        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.repetition_penalty, 1.3)
        self.assertEqual(args.trace_top_tokens, 3)
        self.assertTrue(args.use_kv_cache)
        self.assertEqual(args.samples_jsonl, Path("samples.jsonl"))

    def test_direct_answer_eval_reports_strict_exact_without_candidates(self) -> None:
        record = {
            "id": "one",
            "prompt": "q:\na:",
            "target": " a.",
        }
        tokenizer = CharTokenizer.train(
            record["prompt"] + record["target"] + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=25,
            )
        )
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source="qa:color",
        )
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(4)
        for _ in range(120):
            train_direct_answer_first_error(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.12,
                terminator=ANSWER_TERMINATOR,
            )

        result = evaluate_direct_answer_records(
            model,
            tokenizer,
            [record],
            max_new_chars=16,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["failed_records"], [])

    def test_answer_eval_can_use_candidate_selector(self) -> None:
        records = [
            {
                "id": "one",
                "prompt": "question: what color is mia's ring?\nanswer:",
                "target": " green.",
            }
        ]
        examples = [
            AnswerExample(
                prompt=records[0]["prompt"],
                target=" green.",
                source="qa:color",
            ),
            AnswerExample(
                prompt="question: where is mia's ring?\nanswer:",
                target=" in the box.",
                source="qa:place",
            ),
        ]
        selector = build_answer_selector(examples, seed=22)
        for _ in range(80):
            for example in examples:
                selector.train_step(
                    example,
                    learning_rate=0.08,
                    candidates=[" green.", " in the box."],
                )
        tokenizer = CharTokenizer.train(
            records[0]["prompt"] + " green." + " in the box."
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=23,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=[" in the box.", " green."],
            max_new_chars=12,
            include_completions=False,
            selector=selector,
        )

        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_candidate_records"], [])

    def test_answer_eval_can_emit_selector_choice_as_completion(self) -> None:
        records = [
            {
                "id": "one",
                "prompt": "question: what color is mia's ring?\nanswer:",
                "target": " green.",
            }
        ]
        examples = [
            AnswerExample(
                prompt=records[0]["prompt"],
                target=" green.",
                source="qa:color",
            ),
            AnswerExample(
                prompt="question: where is mia's ring?\nanswer:",
                target=" in the box.",
                source="qa:place",
            ),
        ]
        selector = build_answer_selector(examples, seed=24)
        candidates = [" in the box.", " green."]
        for _ in range(80):
            for example in examples:
                selector.train_step(example, learning_rate=0.08, candidates=candidates)
        tokenizer = CharTokenizer.train(records[0]["prompt"] + "".join(candidates))
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=25,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=candidates,
            max_new_chars=12,
            include_completions=False,
            selector=selector,
            emit_selected_candidate=True,
        )

        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_records"], [])
        self.assertEqual(result["failed_candidate_records"], [])

    def test_transformer_guided_generator_learns_without_candidates(self) -> None:
        examples = [
            AnswerExample(
                prompt="question: what color is mia's ring?\nanswer:",
                target=" green.",
                source="qa:color",
            ),
            AnswerExample(
                prompt="question: where is mia's ring?\nanswer:",
                target=" in the box.",
                source="qa:place",
            ),
        ]
        tokenizer = CharTokenizer.train(
            "".join(example.prompt + example.target for example in examples)
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=26,
            )
        )
        generator = build_transformer_answer_generator(
            examples,
            model,
            tokenizer,
            seed=27,
            max_answer_chars=24,
            transformer_top_k=2,
        )

        before = generator.sequence_loss(
            model,
            tokenizer,
            examples[0].prompt,
            examples[0].target,
        )
        for _ in range(180):
            for example in examples:
                generator.train_example(model, tokenizer, example, learning_rate=0.08)
        after = generator.sequence_loss(
            model,
            tokenizer,
            examples[0].prompt,
            examples[0].target,
        )

        self.assertIsInstance(generator, TransformerGuidedAnswerGenerator)
        self.assertGreater(before, after)
        self.assertEqual(generator.generate(model, tokenizer, examples[0].prompt), " green.")
        self.assertEqual(generator.generate(model, tokenizer, examples[1].prompt), " in the box.")

    def test_answer_generator_eval_reports_exact_without_candidates(self) -> None:
        examples = [
            AnswerExample(
                prompt="question: what color is mia's ring?\nanswer:",
                target=" green.",
                source="qa:color",
            )
        ]
        records = [
            {
                "id": "one",
                "prompt": examples[0].prompt,
                "target": examples[0].target,
            }
        ]
        tokenizer = CharTokenizer.train(examples[0].prompt + examples[0].target)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=28,
            )
        )
        generator = build_transformer_answer_generator(
            examples,
            model,
            tokenizer,
            seed=29,
            max_answer_chars=24,
            transformer_top_k=2,
        )
        for _ in range(180):
            generator.train_example(model, tokenizer, examples[0], learning_rate=0.08)

        result = evaluate_answer_generator_records(generator, model, tokenizer, records)

        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["failed_records"], [])


if __name__ == "__main__":
    unittest.main()
