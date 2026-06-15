from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.neural_char_model import context_before, continuation_nll
from closed_world_lm.tokenizer import CharTokenizer
from closed_world_lm.answer_model import AnswerExample
from closed_world_lm.transformer_char_model import (
    AnswerCandidateSelector,
    GenerationConfig,
    OptimizationConfig,
    ScalarOptimizer,
    TransformerGuidedAnswerGenerator,
    TinyTransformerLM,
    TransformerConfig,
    ANSWER_TERMINATOR,
    answer_sequence_nll,
    build_answer_selector,
    build_transformer_answer_generator,
    has_repeated_suffix,
    direct_answer_sequence_nll,
    direct_answer_first_error,
    direct_answer_branch_profile,
    direct_answer_rollout_error,
    direct_answer_early_stop_error,
    direct_answer_repeat_loop_error,
    direct_answer_generated_prefix_recovery,
    direct_answer_sequence_repair_errors,
    direct_answer_branch_repair_error,
    direct_answer_branch_context,
    direct_answer_branch_target_ids,
    direct_answer_branch_representation_profile,
    direct_answer_branch_span_position,
    direct_answer_branch_span_repair_error,
    direct_answer_branch_batch,
    direct_answer_target_balanced_branch_batch,
    direct_answer_branch_diversity_batch,
    direct_answer_target_balanced_branch_diversity_batch,
    direct_answer_profiled_replay_records,
    direct_answer_dominant_branch_prediction,
    direct_answer_hard_branch_contrast,
    branch_replay_plan,
    audit_prompt_context_coverage,
    audit_direct_answer_branch_context_coverage,
    summarize_branch_context_coverage_gate,
    summarize_branch_diversity_target,
    branch_diversity_snapshot_score,
    branch_diversity_snapshot_preserves_target_coverage,
    evaluate_direct_answer_records,
    evaluate_answer_generator_records,
    evaluate_answer_records,
    generation_distribution,
    score_transformer_records,
    exclude_scalars,
    save_optimizer_state,
    load_optimizer_state,
    sampled_choice_candidates,
    train_direct_answer_first_error,
    train_direct_answer_first_error_unlikelihood,
    train_direct_answer_rollout_unlikelihood,
    train_direct_answer_early_stop_unlikelihood,
    train_direct_answer_repeat_loop_unlikelihood,
    train_direct_answer_balanced_repair_unlikelihood,
    train_direct_answer_generated_prefix_recovery_unlikelihood,
    train_direct_answer_sequence_repair_unlikelihood,
    train_direct_answer_loop_escape_unlikelihood,
    train_direct_answer_branch_repair_unlikelihood,
    train_direct_answer_branch_collapse_unlikelihood,
    train_direct_answer_branch_batch_contrast_unlikelihood,
    train_direct_answer_branch_diversity_unlikelihood,
    train_direct_answer_branch_target_softmax_unlikelihood,
    train_direct_answer_branch_target_margin_unlikelihood,
    train_direct_answer_branch_representation_contrast_unlikelihood,
    train_direct_answer_branch_output_binding_unlikelihood,
    train_direct_answer_branch_bidirectional_binding_unlikelihood,
    train_direct_answer_branch_coverage_binding_unlikelihood,
    train_direct_answer_branch_target_set_coverage_unlikelihood,
    train_direct_answer_branch_target_diversity_unlikelihood,
    train_direct_answer_branch_target_replay_coverage_unlikelihood,
    train_direct_answer_branch_context_replay_coverage_unlikelihood,
    train_direct_answer_branch_rank_margin_unlikelihood,
    train_direct_answer_branch_topk_softmax_unlikelihood,
    train_direct_answer_branch_contrast_unlikelihood,
    train_direct_answer_branch_span_repair_unlikelihood,
    train_direct_answer_branch_span_contrast_unlikelihood,
    train_direct_answer_hard_branch_contrast_unlikelihood,
    train_direct_answer_lesson,
    train_answer_char,
    train_answer_mixed_step,
    direct_answer_lesson,
    flatten_scalars,
    parse_args,
    transformer_experiment_decision,
    transformer_experiment_intent,
    transformer_training_recipe,
    transformer_training_recipe_id,
    transformer_direct_answer_training_pool,
    transformer_answer_generator_training_pool,
)


class TransformerCharModelTest(unittest.TestCase):
    def test_train_step_updates_random_transformer_weights(self) -> None:
        text = "question: where is mia's ball?\nanswer: under the box.\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=6,
            embedding_dim=4,
            feedforward_dim=8,
            seed=3,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 12, config.context_size, tokenizer.pad_id)
        target = ids[12]
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.04)
        after = model.nll(context, target)

        self.assertGreater(before, after)

    def test_adamw_optimizer_accumulates_gradients_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=53,
        )
        model = TinyTransformerLM.init_random(config)
        optimizer = ScalarOptimizer(
            OptimizationConfig(
                optimizer="adamw",
                gradient_accumulation_steps=2,
                warmup_steps=2,
                decay_steps=2,
                min_learning_rate=0.001,
            )
        )
        model.active_optimizer = optimizer
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)

        model.train_step(context, target, learning_rate=0.02)
        self.assertEqual(optimizer.update_count, 0)
        self.assertEqual(optimizer.pending_accumulation, 1)
        model.train_step(context, target, learning_rate=0.02)
        self.assertEqual(optimizer.update_count, 1)
        self.assertEqual(optimizer.pending_accumulation, 0)
        self.assertGreater(before, model.nll(context, target))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "optimizer.json"
            save_optimizer_state(path, optimizer)
            loaded = load_optimizer_state(path, optimizer.config)

        self.assertEqual(loaded.update_count, optimizer.update_count)
        self.assertEqual(loaded.config.optimizer, "adamw")
        self.assertEqual(len(loaded.first_moment), len(optimizer.first_moment))

    def test_v051_architecture_options_forward_and_round_trip(self) -> None:
        text = "abcd abcd\n"
        tokenizer = CharTokenizer.train(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=54,
            attention_heads=2,
            use_pre_layer_norm=True,
            use_rms_norm=True,
            use_gated_mlp=True,
            tie_output_embeddings=True,
            use_rotary_positions=True,
            use_kv_cache_path=True,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(tokenizer.encode(text), 4, config.context_size, tokenizer.pad_id)
        probs = model.predict(context)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer, {"test": "v0.51"})
            loaded, loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(loaded.config.tie_output_embeddings)
        self.assertEqual(loaded.config.attention_heads, 2)
        self.assertIsNotNone(loaded_tokenizer)

    def test_generation_controls_emit_trace_and_cache_metadata(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=55,
                use_kv_cache_path=True,
            )
        )
        generation = model.generate_with_trace(
            tokenizer,
            "abc",
            3,
            GenerationConfig(
                temperature=0.7,
                top_k=2,
                top_p=0.9,
                repetition_penalty=1.1,
                trace_top_tokens=2,
                use_kv_cache=True,
            ),
        )

        self.assertLessEqual(len(generation["trace"]), 3)
        self.assertTrue(generation["cache"]["enabled"])
        self.assertLessEqual(len(generation["trace"][0]["top_tokens"]), 2)

    def test_generation_distribution_applies_top_k_and_repetition_penalty(self) -> None:
        probs = generation_distribution(
            [0.6, 0.3, 0.1],
            [0],
            GenerationConfig(top_k=2, repetition_penalty=3.0),
        )

        self.assertEqual(probs[2], 0.0)
        self.assertGreater(probs[1], probs[0])
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_transformer_eval_scoring_returns_replayable_trace_records(self) -> None:
        example = {"id": "one", "prompt": "q:\na:", "target": " a."}
        text = example["prompt"] + example["target"] + ANSWER_TERMINATOR
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=56,
            )
        )

        records = score_transformer_records(
            model,
            tokenizer,
            [example],
            max_new_chars=2,
            generation_config=GenerationConfig(trace_top_tokens=2),
            candidates=[example["target"]],
        )

        self.assertEqual(records[0]["id"], "one")
        self.assertIn("generation_trace", records[0])
        self.assertIn("candidate_scores", records[0])

    def test_checkpoint_round_trip_includes_corpus_tokenizer(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=1,
        )
        model = TinyTransformerLM.init_random(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertIsNotNone(loaded_tokenizer)
        self.assertEqual(loaded.config.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded_tokenizer.tokens, tokenizer.tokens)  # type: ignore[union-attr]

    def test_layer_normalized_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=9,
            use_layer_norm=True,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        for _ in range(30):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_layer_norm)
        self.assertIn("ln1_gain", weights)
        self.assertIn("ln2_gain", weights)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_pre_layer_normalized_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=11,
            use_pre_layer_norm=True,
        )
        model = TinyTransformerLM.init_random(config)
        baseline = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=11,
            )
        )
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        params = model.parameters()

        self.assertNotEqual(baseline.final_hidden(context), model.final_hidden(context))
        self.assertTrue(any(param is model.ln1_gain[0] for param in params))
        self.assertTrue(any(param is model.ln2_gain[0] for param in params))
        self.assertTrue(any(param is model.final_ln_gain[0] for param in params))

        for _ in range(30):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_pre_layer_norm)
        self.assertIn("final_ln_gain", weights)
        self.assertIn("final_ln_bias", weights)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_pre_layer_norm_scalar_and_float_forward_match(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=12,
                num_layers=2,
                use_pre_layer_norm=True,
            )
        )
        context = context_before(ids, 4, model.config.context_size, tokenizer.pad_id)
        scalar_logits = [value.data for value in model._forward_scalars(context)]
        float_logits = model._forward_floats(context)

        for expected, actual in zip(float_logits, scalar_logits):
            self.assertAlmostEqual(expected, actual)

    def test_legacy_transformer_checkpoint_defaults_pre_layer_norm_off(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=13,
            )
        )
        payload = model.to_dict(tokenizer)
        payload["config"].pop("use_pre_layer_norm", None)
        payload["weights"].pop("final_ln_gain", None)
        payload["weights"].pop("final_ln_bias", None)

        loaded, _loaded_tokenizer = TinyTransformerLM.from_dict(payload)

        self.assertFalse(loaded.config.use_pre_layer_norm)
        self.assertEqual(
            [value.data for value in loaded.final_ln_gain],
            [1.0 for _ in range(loaded.config.embedding_dim)],
        )

    def test_multi_layer_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=10,
            num_layers=2,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertEqual(loaded.config.num_layers, 2)
        self.assertEqual(len(weights["extra_layers"]), 1)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_context_mean_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=13,
            use_context_mean=True,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertTrue(loaded.config.use_context_mean)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_context_projection_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=14,
        )
        projection_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=14,
            use_context_projection=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(projection_config)
        context = context_before(ids, 4, projection_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.context_projection_w)
                + flatten_scalars(model.context_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_context_projection)
        self.assertIn("context_projection_w", weights)
        self.assertIn("context_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_prefix_projection_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=16,
        )
        prefix_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=16,
            use_prompt_prefix_projection=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(prefix_config)
        context = context_before(ids, 4, prefix_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_prefix_projection_w)
                + flatten_scalars(model.prompt_prefix_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_prefix_projection)
        self.assertIn("prompt_prefix_projection_w", weights)
        self.assertIn("prompt_prefix_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_position_projection_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=18,
        )
        position_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=18,
            use_prompt_position_projection=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(position_config)
        context = context_before(ids, 4, position_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_position_projection_w)
                + flatten_scalars(model.prompt_position_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_position_projection)
        self.assertEqual(loaded.config.prompt_position_projection_scale, 1.0)
        self.assertIn("prompt_position_projection_w", weights)
        self.assertIn("prompt_position_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_position_projection_scale_changes_nonzero_residual(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=19,
            use_prompt_position_projection=True,
        )
        scaled_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=19,
            use_prompt_position_projection=True,
            prompt_position_projection_scale=3.0,
        )
        base = TinyTransformerLM.init_random(base_config)
        scaled = TinyTransformerLM.init_random(scaled_config)
        base.prompt_position_projection_b[0].data = 0.25
        scaled.prompt_position_projection_b[0].data = 0.25
        context = context_before(ids, 4, base_config.context_size, tokenizer.pad_id)

        base_hidden = base.final_hidden(context)
        scaled_hidden = scaled.final_hidden(context)
        base_probs = base.predict(context)
        scaled_probs = scaled.predict(context)

        self.assertNotEqual(base_hidden, scaled_hidden)
        self.assertNotEqual(base_probs, scaled_probs)

    def test_final_hidden_matches_forward_logits(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=19,
                use_prompt_position_projection=True,
            )
        )
        context = context_before(ids, 4, model.config.context_size, tokenizer.pad_id)

        hidden = model.final_hidden(context)
        expected_logits = model._forward_floats(context)
        actual_logits = []
        for output_index, bias in enumerate(model.bout):
            total = bias.data
            for input_index, value in enumerate(hidden):
                total += value * model.wout[input_index][output_index].data
            actual_logits.append(total)

        self.assertEqual(len(hidden), model.config.embedding_dim)
        for expected, actual in zip(expected_logits, actual_logits):
            self.assertAlmostEqual(expected, actual)

    def test_prompt_attention_summary_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=15,
        )
        summary_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=15,
            use_prompt_attention_summary=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(summary_config)
        context = context_before(ids, 4, summary_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_summary_w)
                + flatten_scalars(model.prompt_summary_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_attention_summary)
        self.assertIn("prompt_summary_query", weights)
        self.assertIn("prompt_summary_w", weights)
        self.assertIn("prompt_summary_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_multi_layer_top_layer_update_freezes_lower_layer(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=12,
            num_layers=2,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        lower_before = model.wq[0][0].data
        top_before = model.extra_blocks[0]["wq"][0][0].data
        head_before = model.wout[0][target].data
        model.freeze_lower_layers_for_updates = True

        for _ in range(20):
            model.train_step(
                context,
                target,
                learning_rate=0.02,
                params=model.top_layer_parameters(),
            )

        self.assertEqual(model.wq[0][0].data, lower_before)
        self.assertTrue(
            model.extra_blocks[0]["wq"][0][0].data != top_before
            or model.wout[0][target].data != head_before
        )

    def test_multi_layer_final_block_matches_full_stack_logits(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=11,
            num_layers=2,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(tokenizer.encode(text), 4, config.context_size, tokenizer.pad_id)
        optimized = model._forward_floats(context)
        token_embeddings = [[value.data for value in row] for row in model.token_embeddings]
        position_embeddings = [[value.data for value in row] for row in model.position_embeddings]
        full_stack = model._forward_full_block_floats(
            [
                [
                    token_embeddings[token_id][dim] + position_embeddings[position][dim]
                    for dim in range(config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ],
            model._block_to_floats(model.blocks[0]),
        )
        full_stack = model._forward_full_block_floats(
            full_stack,
            model._block_to_floats(model.blocks[1]),
        )
        manual = []
        for output_index, bias in enumerate([value.data for value in model.bout]):
            total = bias
            for input_index, value in enumerate(full_stack[-1]):
                total += value * model.wout[input_index][output_index].data
            manual.append(total)

        self.assertEqual(len(optimized), len(manual))
        for left, right in zip(optimized, manual):
            self.assertAlmostEqual(left, right)

    def test_generate_uses_character_tokenizer(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=5,
            )
        )

        generated = model.generate(tokenizer, "abc", max_new_chars=3)

        self.assertIsInstance(generated, str)
        self.assertLessEqual(len(generated), 3)

    def test_generate_can_stop_at_admitted_terminator(self) -> None:
        text = "a\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=2,
                embedding_dim=4,
                feedforward_dim=8,
                seed=6,
            )
        )
        newline_id = tokenizer.stoi[ANSWER_TERMINATOR]
        model.bout[newline_id].data = 5.0

        generated = model.generate(
            tokenizer,
            "a",
            max_new_chars=4,
            stop_at=ANSWER_TERMINATOR,
        )

        self.assertEqual(generated, "")

    def test_answer_lesson_training_reduces_continuation_loss(self) -> None:
        example = AnswerExample(
            prompt="question: where is mia's ring?\nanswer:",
            target=" in the box.",
            source="qa:place",
        )
        tokenizer = CharTokenizer.train(example.prompt + example.target)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=9,
            )
        )
        before = answer_sequence_nll(model, tokenizer, example)
        rng = random.Random(1)
        for _ in range(80):
            train_answer_char(model, tokenizer, example, rng, learning_rate=0.05)
        after = answer_sequence_nll(model, tokenizer, example)

        self.assertGreater(before, after)

    def test_answer_eval_reports_candidate_matches(self) -> None:
        records = [
            {
                "id": "one",
                "prompt": "question: where is mia's ring?\nanswer:",
                "target": " in the box.",
            }
        ]
        tokenizer = CharTokenizer.train(records[0]["prompt"] + records[0]["target"])
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=11,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=[" in the box."],
            max_new_chars=12,
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_candidate_records"], [])

    def test_answer_eval_can_skip_slow_completions(self) -> None:
        records = [
            {
                "id": "one",
                "prompt": "question: where is mia's ring?\nanswer:",
                "target": " in the box.",
            }
        ]
        tokenizer = CharTokenizer.train(records[0]["prompt"] + records[0]["target"])
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=12,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=[" in the box."],
            max_new_chars=12,
            include_completions=False,
        )

        self.assertIsNone(result["exact"])
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_records"], [])

    def test_sampled_choice_candidates_keeps_target_first(self) -> None:
        rng = random.Random(4)

        candidates = sampled_choice_candidates(
            " green.",
            [" red.", " green.", " blue.", " red."],
            rng,
            negative_count=1,
        )

        self.assertEqual(candidates[0], " green.")
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[1], " green.")

    def test_mixed_answer_training_improves_candidate_margin(self) -> None:
        example = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        negative = " red."
        tokenizer = CharTokenizer.train(example.prompt + example.target + negative)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=13,
            )
        )
        rng = random.Random(2)

        before_margin = continuation_nll(
            model,
            tokenizer,
            example.prompt,
            example.target,
        ) - continuation_nll(model, tokenizer, example.prompt, negative)
        for _ in range(60):
            train_answer_mixed_step(
                model,
                tokenizer,
                example,
                rng,
                learning_rate=0.05,
                candidates=[example.target, negative],
                target_loss_weight=1.0,
                choice_loss_weight=1.0,
                choice_negatives=1,
                choice_max_chars=4,
            )
        after_margin = continuation_nll(
            model,
            tokenizer,
            example.prompt,
            example.target,
        ) - continuation_nll(model, tokenizer, example.prompt, negative)

        self.assertLess(after_margin, before_margin)

    def test_answer_candidate_selector_learns_from_closed_world_examples(self) -> None:
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
        selector = build_answer_selector(examples, seed=21)
        candidates = [" green.", " in the box."]

        before = selector.loss(examples[0].prompt, examples[0].target, candidates)
        for _ in range(80):
            for example in examples:
                selector.train_step(example, learning_rate=0.08, candidates=candidates)
        after = selector.loss(examples[0].prompt, examples[0].target, candidates)

        self.assertIsInstance(selector, AnswerCandidateSelector)
        self.assertGreater(before, after)
        self.assertEqual(selector.predict(examples[0].prompt, candidates), " green.")
        self.assertEqual(selector.predict(examples[1].prompt, candidates), " in the box.")

    def test_transformer_generator_pool_prioritizes_long_operational_lessons(self) -> None:
        fact = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        learning = AnswerExample(
            prompt="question: how do you improve?\nanswer:",
            target=" by admitted training data.",
            source="qa:learning",
        )

        pool = transformer_answer_generator_training_pool([fact, learning])

        self.assertGreater(pool.count(learning), pool.count(fact))

    def test_direct_answer_pool_prioritizes_long_operational_lessons(self) -> None:
        fact = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        learning = AnswerExample(
            prompt="question: how do you improve?\nanswer:",
            target=" by admitted training data.",
            source="qa:learning",
        )

        pool = transformer_direct_answer_training_pool([fact, learning])

        self.assertGreater(pool.count(learning), pool.count(fact))

    def test_direct_answer_training_updates_transformer_without_candidates(self) -> None:
        example = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=24,
            )
        )
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(3)

        before = direct_answer_sequence_nll(model, tokenizer, example, ANSWER_TERMINATOR)
        for _ in range(120):
            train_direct_answer_lesson(model, lesson, rng, learning_rate=0.05)
        after = direct_answer_sequence_nll(model, tokenizer, example, ANSWER_TERMINATOR)

        self.assertGreater(before, after)

    def test_direct_answer_first_error_targets_greedy_mismatch(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" a.", source="qa:color")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=24,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        repair = direct_answer_first_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(repair)
        _context, target_id, predicted_id, position = repair  # type: ignore[misc]
        self.assertEqual(tokenizer.itos[target_id], " ")
        self.assertEqual(tokenizer.itos[predicted_id], ".")
        self.assertEqual(position, 0)

    def test_direct_answer_branch_profile_summarizes_branch_confusion(self) -> None:
        record = {
            "id": "color",
            "prompt": "q:\na:",
            "target": " a.",
        }
        tokenizer = CharTokenizer.train(record["prompt"] + record["target"] + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=24,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        profile = direct_answer_branch_profile(
            model,
            tokenizer,
            [record],
            branch_position=0,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 1)
        self.assertEqual(profile["correct"], 0)
        self.assertEqual(profile["skipped"], 0)
        self.assertLess(profile["avg_target_margin"], 0.0)
        self.assertGreater(profile["target_rank"]["avg"], 1.0)
        self.assertEqual(profile["target_rank"]["top1_rate"], 0.0)
        self.assertEqual(profile["target_rank"]["vocab_size"], tokenizer.vocab_size)
        self.assertEqual(profile["target_tokens"][0], {"value": " ", "count": 1})
        self.assertEqual(profile["predicted_tokens"][0], {"value": ".", "count": 1})
        self.assertEqual(profile["confusions"][0], {"value": "' '->'.'", "count": 1})
        self.assertEqual(profile["failed_records"][0]["id"], "color")
        self.assertEqual(profile["failed_records"][0]["target_token"], " ")
        self.assertEqual(profile["failed_records"][0]["predicted_token"], ".")
        self.assertGreaterEqual(profile["failed_records"][0]["target_rank"], 2)
        self.assertEqual(
            profile["failed_records"][0]["top_predictions"][0]["token"],
            ".",
        )

    def test_direct_answer_branch_profile_reports_diversity_collapse(self) -> None:
        records = [
            {"id": "near", "prompt": "q: where?\na:", "target": " near."},
            {"id": "green", "prompt": "q: color?\na:", "target": " green."},
        ]
        tokenizer = CharTokenizer.train(
            "".join(record["prompt"] + record["target"] for record in records)
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=43,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        profile = direct_answer_branch_profile(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["diversity"]["target_unique"], 2)
        self.assertEqual(profile["diversity"]["predicted_unique"], 1)
        self.assertEqual(profile["diversity"]["target_token_coverage"], 0.0)
        self.assertEqual(profile["diversity"]["dominant_predicted_token"], ".")
        self.assertEqual(profile["diversity"]["dominant_predicted_count"], 2)
        self.assertEqual(profile["diversity"]["dominant_predicted_rate"], 1.0)
        self.assertTrue(profile["diversity"]["collapsed"])
        self.assertEqual(
            profile["diversity"]["missing_target_tokens"],
            [{"value": "n", "count": 1}, {"value": "g", "count": 1}],
        )

    def test_branch_diversity_target_marks_collapsed_profiles(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.5,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                        "missing_target_tokens": [{"value": "b", "count": 1}],
                    }
                },
                "self": {
                    "diversity": {
                        "target_unique": 1,
                        "predicted_unique": 1,
                        "target_token_coverage": 1.0,
                        "dominant_predicted_token": "s",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": False,
                        "missing_target_tokens": [],
                    }
                },
            }
        )

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["multi_target_profiles"], 1)
        self.assertEqual(summary["passed_profiles"], 0)
        self.assertEqual(summary["failed_profiles"], 1)
        self.assertEqual(summary["max_dominant_predicted_rate"], 1.0)
        self.assertEqual(summary["min_target_token_coverage"], 0.5)
        self.assertEqual(summary["blocking_evals"][0]["name"], "qa")
        self.assertTrue(summary["blocking_evals"][0]["collapsed"])

    def test_branch_diversity_target_passes_when_targets_are_covered(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 2,
                        "target_token_coverage": 1.0,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 0.5,
                        "collapsed": False,
                        "missing_target_tokens": [],
                    }
                }
            }
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["multi_target_profiles"], 1)
        self.assertEqual(summary["passed_profiles"], 1)
        self.assertEqual(summary["failed_profiles"], 0)
        self.assertEqual(summary["blocking_evals"], [])

    def test_branch_diversity_snapshot_score_prefers_more_prediction_diversity(self) -> None:
        collapsed = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    }
                }
            },
        }
        cracked = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 0.75,
                    }
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(cracked),
            branch_diversity_snapshot_score(collapsed),
        )

    def test_branch_diversity_snapshot_score_uses_target_rank_tiebreaker(self) -> None:
        buried = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 20.0,
                        "top3_rate": 0.0,
                        "top5_rate": 0.0,
                    },
                }
            },
        }
        lifted = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 8.0,
                        "top3_rate": 0.25,
                        "top5_rate": 0.5,
                    },
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(lifted),
            branch_diversity_snapshot_score(buried),
        )

    def test_branch_diversity_snapshot_score_prefers_rank_over_wrong_diversity(self) -> None:
        wrong_diverse = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 0.75,
                    },
                    "target_rank": {
                        "avg": 14.0,
                        "top3_rate": 0.0,
                        "top5_rate": 0.25,
                    },
                }
            },
        }
        rank_lifted = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 12.0,
                        "top3_rate": 0.25,
                        "top5_rate": 0.25,
                    },
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(rank_lifted),
            branch_diversity_snapshot_score(wrong_diverse),
        )

    def test_branch_diversity_snapshot_coverage_floor_is_profile_wise(self) -> None:
        baseline = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "self": {
                    "diversity": {
                        "target_unique": 1,
                        "target_token_coverage": 1.0,
                    }
                },
            }
        }
        rank_lifted_but_forgetting = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.0,
                    },
                    "target_rank": {
                        "avg": 4.0,
                        "top3_rate": 0.5,
                        "top5_rate": 0.75,
                    },
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.5,
                    },
                    "target_rank": {
                        "avg": 4.0,
                        "top3_rate": 0.5,
                        "top5_rate": 0.75,
                    },
                },
            }
        }
        coverage_preserved = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
            }
        }

        self.assertFalse(
            branch_diversity_snapshot_preserves_target_coverage(
                rank_lifted_but_forgetting,
                baseline,
            )
        )
        self.assertTrue(
            branch_diversity_snapshot_preserves_target_coverage(
                coverage_preserved,
                baseline,
            )
        )

    def test_branch_context_coverage_marks_truncated_semantic_branch(self) -> None:
        record = {
            "id": "place",
            "prompt": "question: where is mia's ball?\nanswer:",
            "target": " under.",
        }
        tokenizer = CharTokenizer.train(record["prompt"] + record["target"] + ANSWER_TERMINATOR)
        narrow = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=3,
                feedforward_dim=5,
                seed=41,
            )
        )
        wide = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=48,
                embedding_dim=3,
                feedforward_dim=5,
                seed=41,
            )
        )

        narrow_audit = audit_direct_answer_branch_context_coverage(
            narrow,
            tokenizer,
            [record],
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        wide_audit = audit_direct_answer_branch_context_coverage(
            wide,
            tokenizer,
            [record],
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(narrow_audit["semantic_records"], 1)
        self.assertEqual(narrow_audit["missing"], 1)
        self.assertIn("intent:place", narrow_audit["missing_records"][0]["missing_features"])
        self.assertEqual(wide_audit["covered"], 1)
        self.assertEqual(wide_audit["missing_records"], [])

    def test_branch_context_coverage_marks_ambiguous_context_collisions(self) -> None:
        records = [
            {"id": "one", "prompt": "q: one\na:", "target": " red."},
            {"id": "two", "prompt": "q: two\na:", "target": " blue."},
        ]
        text = "".join(record["prompt"] + record["target"] for record in records)
        tokenizer = CharTokenizer.train(text + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=3,
                embedding_dim=3,
                feedforward_dim=5,
                seed=42,
            )
        )

        audit = audit_direct_answer_branch_context_coverage(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(audit["count"], 2)
        self.assertEqual(audit["unique_contexts"], 1)
        self.assertEqual(audit["collision_contexts"], 1)
        self.assertEqual(audit["ambiguous_contexts"], 1)
        self.assertEqual(audit["max_context_reuse"], 2)
        self.assertEqual(audit["max_target_options"], 2)
        self.assertEqual(audit["ambiguous_records"][0]["context_text"], "a: ")
        self.assertEqual(
            audit["ambiguous_records"][0]["target_tokens"],
            [{"value": "r", "count": 1}, {"value": "b", "count": 1}],
        )

    def test_branch_context_coverage_gate_summarizes_blockers(self) -> None:
        summary = summarize_branch_context_coverage_gate(
            {
                "qa": {
                    "count": 2,
                    "semantic_records": 2,
                    "covered": 1,
                    "missing": 1,
                    "covered_rate": 0.5,
                    "ambiguous_contexts": 1,
                    "collision_contexts": 1,
                    "skipped": 0,
                },
                "self": {
                    "count": 1,
                    "semantic_records": 1,
                    "covered": 1,
                    "missing": 0,
                    "covered_rate": 1.0,
                    "ambiguous_contexts": 0,
                    "collision_contexts": 0,
                    "skipped": 0,
                },
            }
        )

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["covered"], 2)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["ambiguous_contexts"], 1)
        self.assertEqual(summary["blocking_evals"][0]["name"], "qa")

    def test_dominant_branch_prediction_finds_global_wrong_token(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=38,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        dominant = direct_answer_dominant_branch_prediction(
            model,
            tokenizer,
            [near, green],
            random.Random(8),
            branch_position=1,
            sample_count=0,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(dominant)
        predicted_id, count, scored = dominant  # type: ignore[misc]
        self.assertEqual(tokenizer.itos[predicted_id], ".")
        self.assertEqual(count, 2)
        self.assertEqual(scored, 2)

    def test_branch_collapse_repair_penalizes_dominant_wrong_token(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=39,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            near,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(branch)
        near_context, near_target, _position = branch  # type: ignore[misc]
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        before_wrong = model.predict(near_context)[wrong_id]
        before_target = model.predict(near_context)[near_target]
        rng = random.Random(9)

        for _ in range(32):
            train_direct_answer_branch_collapse_unlikelihood(
                model,
                tokenizer,
                near,
                [near, green],
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                sample_count=0,
                terminator=ANSWER_TERMINATOR,
            )

        after_probs = model.predict(near_context)
        self.assertLess(after_probs[wrong_id], before_wrong)
        self.assertGreater(after_probs[near_target], before_target)

    def test_branch_batch_selects_distinct_branch_targets(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=40,
            )
        )

        batch = direct_answer_branch_batch(
            model,
            tokenizer,
            near,
            [near, green, tree],
            random.Random(11),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {tokenizer.itos[target] for _context, target in batch},
            {"n", "g", "t"},
        )

    def test_target_balanced_branch_batch_samples_rare_targets(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        blue = AnswerExample(prompt="q: blue?\na:", target=" blue.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + blue.prompt
            + blue.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=40,
            )
        )
        skewed_examples = [near for _ in range(20)] + [green, tree, blue]

        batch = direct_answer_target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            skewed_examples,
            random.Random(11),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )
        diversity_batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            skewed_examples,
            random.Random(11),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 4)
        self.assertEqual(
            {tokenizer.itos[target] for _context, target in batch},
            {"n", "g", "t", "b"},
        )
        self.assertEqual(
            {tokenizer.itos[target] for _context, target, _predicted in diversity_batch},
            {"n", "g", "t", "b"},
        )

    def test_branch_diversity_batch_records_current_predictions(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=44,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            [near, green, tree],
            random.Random(14),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {tokenizer.itos[target] for _context, target, _predicted in batch},
            {"n", "g", "t"},
        )
        self.assertEqual(
            {tokenizer.itos[predicted] for _context, _target, predicted in batch},
            {"."},
        )

    def test_branch_batch_contrast_improves_prompt_branch_margin(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=41,
            )
        )
        batch = direct_answer_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(12),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = {target for _context, target in batch}

        def branch_margin() -> float:
            total = 0.0
            for context, target in batch:
                probs = model.predict(context)
                strongest_other = max(
                    probs[other]
                    for other in branch_targets
                    if other != target
                )
                total += probs[target] - strongest_other
            return total

        before = branch_margin()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(13)

        for _ in range(64):
            train_direct_answer_branch_batch_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = branch_margin()
        self.assertGreater(after, before)

    def test_branch_diversity_unlikelihood_suppresses_global_wrong_token(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=45,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def batch_scores() -> tuple[float, float]:
            wrong_total = 0.0
            target_total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                wrong_total += probs[wrong_id]
                target_total += probs[target]
            return wrong_total, target_total

        before_wrong, before_target = batch_scores()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_diversity_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                contrast_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after_wrong, after_target = batch_scores()
        self.assertLess(after_wrong, before_wrong)
        self.assertGreater(after_target, before_target)

    def test_branch_diversity_training_can_freeze_output_bias(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=45,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        params = exclude_scalars(model.parameters(), model.bout)
        before_bout = [value.data for value in model.bout]
        before_wout = model.wout[0][wrong_id].data
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )

        train_direct_answer_branch_diversity_unlikelihood(
            model,
            tokenizer,
            near,
            examples,
            lesson,
            random.Random(16),
            learning_rate=0.06,
            negative_weight=1.0,
            positive_weight=1.0,
            contrast_weight=1.0,
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
            params=params,
        )

        self.assertEqual([value.data for value in model.bout], before_bout)
        self.assertNotEqual(model.wout[0][wrong_id].data, before_wout)

    def test_branch_target_softmax_improves_restricted_branch_choice(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=45,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})

        def restricted_target_probability() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                denominator = sum(probs[branch_target] for branch_target in branch_targets)
                total += probs[target] / denominator
            return total

        before = restricted_target_probability()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_softmax_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                target_softmax_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = restricted_target_probability()
        self.assertGreater(after, before)

    def test_branch_target_margin_improves_restricted_logit_gap(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=42,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})

        def restricted_logit_gap() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                logits = model._forward_floats(context)
                strongest_other = max(
                    logits[other]
                    for other in branch_targets
                    if other != target
                )
                total += logits[target] - strongest_other
            return total

        before = restricted_logit_gap()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_margin_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.02,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = restricted_logit_gap()
        self.assertGreater(after, before)

    def test_branch_representation_profile_reports_hidden_distances(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        records = [
            {"id": "near", "prompt": near.prompt, "target": near.target},
            {"id": "green", "prompt": green.prompt, "target": green.target},
            {"id": "tree", "prompt": tree.prompt, "target": tree.target},
        ]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=47,
            )
        )

        profile = direct_answer_branch_representation_profile(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 3)
        self.assertEqual(profile["skipped"], 0)
        self.assertEqual(profile["target_unique"], 3)
        self.assertEqual(profile["different_target_pairwise_distance"]["count"], 3)
        self.assertGreater(profile["different_target_pairwise_distance"]["avg"], 0.0)

    def test_branch_representation_contrast_increases_hidden_distance(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=48,
            )
        )
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def average_hidden_distance() -> float:
            distances = []
            for left_index, (left_context, left_target, _left_predicted) in enumerate(batch):
                left_hidden = model.final_hidden(left_context)
                for right_context, right_target, _right_predicted in batch[left_index + 1:]:
                    if left_target == right_target:
                        continue
                    right_hidden = model.final_hidden(right_context)
                    distances.append(
                        sum(
                            (left_value - right_value) ** 2
                            for left_value, right_value in zip(left_hidden, right_hidden)
                        )
                    )
            return sum(distances) / len(distances)

        before = average_hidden_distance()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_representation_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                representation_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = average_hidden_distance()
        self.assertGreater(after, before)

    def test_branch_output_binding_improves_rank_and_hidden_distance(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=49,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})

        def restricted_target_probability() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                denominator = sum(probs[branch_target] for branch_target in branch_targets)
                total += probs[target] / denominator
            return total

        def average_hidden_distance() -> float:
            distances = []
            for left_index, (left_context, left_target, _left_predicted) in enumerate(batch):
                left_hidden = model.final_hidden(left_context)
                for right_context, right_target, _right_predicted in batch[left_index + 1:]:
                    if left_target == right_target:
                        continue
                    right_hidden = model.final_hidden(right_context)
                    distances.append(
                        sum(
                            (left_value - right_value) ** 2
                            for left_value, right_value in zip(left_hidden, right_hidden)
                        )
                    )
            return sum(distances) / len(distances)

        before_probability = restricted_target_probability()
        before_distance = average_hidden_distance()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_output_binding_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                binding_weight=2.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(restricted_target_probability(), before_probability)
        self.assertGreater(average_hidden_distance(), before_distance)

    def test_branch_rank_margin_lifts_targets_above_hard_negatives(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=50,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def average_target_rank() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                ranked = sorted(
                    range(len(probs)),
                    key=lambda index: probs[index],
                    reverse=True,
                )
                total += ranked.index(target) + 1
            return total / len(batch)

        before_rank = average_target_rank()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_rank_margin_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertLess(average_target_rank(), before_rank)

    def test_balanced_branch_rank_margin_uses_target_balanced_batches(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=51,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def average_target_rank() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                ranked = sorted(
                    range(len(probs)),
                    key=lambda index: probs[index],
                    reverse=True,
                )
                total += ranked.index(target) + 1
            return total / len(batch)

        before_rank = average_target_rank()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_rank_margin_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertLess(average_target_rank(), before_rank)

    def test_branch_bidirectional_binding_lifts_target_context_ownership(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=53,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})
        self.assertGreater(len(branch_targets), 1)

        def average_target_context_ownership() -> float:
            total = 0.0
            for branch_target in branch_targets:
                target_logits = [
                    model._forward_floats(context)[branch_target]
                    for context, _target, _predicted in batch
                ]
                max_logit = max(target_logits)
                exp_scores = [
                    math.exp(target_logit - max_logit)
                    for target_logit in target_logits
                ]
                denominator = sum(exp_scores)
                owned_mass = 0.0
                for exp_score, (_context, target, _predicted) in zip(exp_scores, batch):
                    if target == branch_target:
                        owned_mass += exp_score / denominator
                total += owned_mass
            return total / len(branch_targets)

        before_ownership = average_target_context_ownership()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_bidirectional_binding_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                binding_weight=2.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertGreater(average_target_context_ownership(), before_ownership)

    def test_branch_coverage_binding_lifts_target_set_against_hard_wrong_tokens(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=54,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})
        branch_target_set = set(branch_targets)
        self.assertGreater(len(branch_targets), 1)

        def restricted_probabilities() -> tuple[float, float]:
            target_set_total = 0.0
            target_total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index not in branch_target_set
                ][:5]
                candidate_ids = [*branch_targets, *hard_candidates]
                denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
                target_set_total += (
                    sum(probs[branch_target] for branch_target in branch_targets)
                    / denominator
                )
                target_total += probs[target] / denominator
            return target_set_total / len(batch), target_total / len(batch)

        before_target_set, before_target = restricted_probabilities()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_coverage_binding_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                binding_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        after_target_set, after_target = restricted_probabilities()
        self.assertGreater(after_target_set, before_target_set)
        self.assertGreater(after_target, before_target)

    def test_branch_target_set_coverage_lifts_set_without_exact_sharpening(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=55,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})
        branch_target_set = set(branch_targets)
        self.assertGreater(len(branch_targets), 1)

        def restricted_target_set_mass() -> float:
            total = 0.0
            for context, _target, _predicted in batch:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index not in branch_target_set
                ][:5]
                candidate_ids = [*branch_targets, *hard_candidates]
                denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
                total += (
                    sum(probs[branch_target] for branch_target in branch_targets)
                    / denominator
                )
            return total / len(batch)

        before_mass = restricted_target_set_mass()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_set_coverage_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                coverage_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertGreater(restricted_target_set_mass(), before_mass)

    def test_branch_target_diversity_lifts_set_and_target_share_balance(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=56,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        model.bout[tokenizer.stoi["n"]].data = 4.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})
        branch_target_set = set(branch_targets)
        self.assertGreater(len(branch_targets), 1)

        def restricted_target_metrics() -> tuple[float, float]:
            target_set_total = 0.0
            target_share_totals = [0.0 for _branch_target in branch_targets]
            for context, _target, _predicted in batch:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index not in branch_target_set
                ][:5]
                candidate_ids = [*branch_targets, *hard_candidates]
                denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
                target_values = [
                    probs[branch_target] / denominator
                    for branch_target in branch_targets
                ]
                target_set_mass = sum(target_values)
                target_set_total += target_set_mass
                for offset, target_value in enumerate(target_values):
                    target_share_totals[offset] += target_value / target_set_mass
            average_target_shares = [
                target_share_total / len(batch)
                for target_share_total in target_share_totals
            ]
            return target_set_total / len(batch), min(average_target_shares)

        before_mass, before_min_share = restricted_target_metrics()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_diversity_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                diversity_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        after_mass, after_min_share = restricted_target_metrics()
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_min_share, before_min_share)

    def test_branch_target_replay_coverage_uses_pool_targets_beyond_batch(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        blue = AnswerExample(prompt="q: thing?\na:", target=" blue.", source="qa:thing")
        examples = [near, green, tree, blue]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + blue.prompt
            + blue.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=57,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        model.bout[tokenizer.stoi["n"]].data = 4.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = direct_answer_branch_target_ids(
            model,
            tokenizer,
            examples,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        batch_target_set = {target for _context, target, _predicted in batch}
        replay_target_set = set(replay_targets)
        missing_targets = replay_target_set - batch_target_set
        self.assertEqual(len(batch_target_set), 2)
        self.assertGreater(len(replay_targets), len(batch_target_set))
        self.assertTrue(missing_targets)

        def replay_target_metrics() -> tuple[float, float]:
            target_set_total = 0.0
            missing_share_totals = [0.0 for _missing_target in missing_targets]
            missing_offsets = [
                offset
                for offset, replay_target in enumerate(replay_targets)
                if replay_target in missing_targets
            ]
            for context, _target, _predicted in batch:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index not in replay_target_set
                ][:5]
                candidate_ids = [*replay_targets, *hard_candidates]
                denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
                target_values = [
                    probs[replay_target] / denominator
                    for replay_target in replay_targets
                ]
                target_set_mass = sum(target_values)
                target_set_total += target_set_mass
                for missing_index, target_offset in enumerate(missing_offsets):
                    missing_share_totals[missing_index] += (
                        target_values[target_offset] / target_set_mass
                    )
            average_missing_shares = [
                missing_share_total / len(batch)
                for missing_share_total in missing_share_totals
            ]
            return target_set_total / len(batch), min(average_missing_shares)

        before_mass, before_missing_share = replay_target_metrics()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(64):
            train_direct_answer_branch_target_replay_coverage_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                branch_position=1,
                batch_size=2,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        after_mass, after_missing_share = replay_target_metrics()
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_missing_share, before_missing_share)

    def test_branch_context_replay_coverage_lifts_owned_replay_targets(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        blue = AnswerExample(prompt="q: thing?\na:", target=" blue.", source="qa:thing")
        examples = [near, green, tree, blue]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + blue.prompt
            + blue.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=58,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        model.bout[tokenizer.stoi["n"]].data = 4.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )
        batch_targets = {target for _context, target, _predicted in batch}
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        self.assertEqual(len(batch_targets), 2)
        self.assertGreater(len(replay_targets), len(batch_targets))

        def replay_context_metrics() -> tuple[float, float]:
            target_set_total = 0.0
            owned_shares = []
            for context, target, _predicted in replay_branches:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index not in replay_target_set
                ][:5]
                candidate_ids = [*replay_targets, *hard_candidates]
                denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
                target_values = [
                    probs[replay_target] / denominator
                    for replay_target in replay_targets
                ]
                target_set_mass = sum(target_values)
                target_set_total += target_set_mass
                target_offset = replay_targets.index(target)
                owned_shares.append(target_values[target_offset] / target_set_mass)
            return target_set_total / len(replay_branches), min(owned_shares)

        before_mass, before_owned_share = replay_context_metrics()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(17)

        for _ in range(80):
            train_direct_answer_branch_context_replay_coverage_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                branch_position=1,
                batch_size=2,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        after_mass, after_owned_share = replay_context_metrics()
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_owned_share, before_owned_share)

    def test_branch_context_coverage_anchor_lifts_covered_target_probability(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )

        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=tokenizer.vocab_size,
                    context_size=8,
                    embedding_dim=4,
                    feedforward_dim=8,
                    seed=59,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        model = initialized_model()
        anchored_model = initialized_model()
        covered_branch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=1,
            terminator=ANSWER_TERMINATOR,
        )[0]
        context, target, predicted = covered_branch
        self.assertEqual(target, predicted)
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)

        def covered_anchor_probability(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict(context)
            hard_candidates = [
                index
                for index in sorted(
                    range(len(probs)),
                    key=lambda item: probs[item],
                    reverse=True,
                )
                if index not in replay_target_set
            ][:5]
            candidate_ids = [*replay_targets, *hard_candidates]
            denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
            return probs[target] / denominator

        before_probability = covered_anchor_probability(model)
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        unanchored_rng = random.Random(17)
        anchored_rng = random.Random(17)

        for _ in range(48):
            train_direct_answer_branch_context_replay_coverage_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                unanchored_rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                branch_position=1,
                batch_size=2,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )
        for _ in range(48):
            train_direct_answer_branch_context_replay_coverage_unlikelihood(
                anchored_model,
                tokenizer,
                near,
                examples,
                lesson,
                anchored_rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                branch_position=1,
                batch_size=2,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
                preserve_covered_targets=True,
            )

        self.assertLess(covered_anchor_probability(model), before_probability)
        self.assertGreater(
            covered_anchor_probability(anchored_model),
            covered_anchor_probability(model),
        )

    def test_branch_context_target_balanced_anchor_skips_single_covered_target(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )

        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=tokenizer.vocab_size,
                    context_size=8,
                    embedding_dim=4,
                    feedforward_dim=8,
                    seed=60,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        unanchored_model = initialized_model()
        global_anchor_model = initialized_model()
        balanced_anchor_model = initialized_model()
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            unanchored_model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        covered_targets = {
            target
            for _context, target, predicted in replay_branches
            if target == predicted
        }
        self.assertEqual(covered_targets, {tokenizer.stoi["n"]})
        branch_batch = direct_answer_target_balanced_branch_diversity_batch(
            unanchored_model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        context, target, _predicted = replay_branches[0]

        def covered_anchor_probability(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict(context)
            hard_candidates = [
                index
                for index in sorted(
                    range(len(probs)),
                    key=lambda item: probs[item],
                    reverse=True,
                )
                if index not in replay_target_set
            ][:5]
            candidate_ids = [*replay_targets, *hard_candidates]
            denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
            return probs[target] / denominator

        before_probability = covered_anchor_probability(unanchored_model)

        for _ in range(48):
            unanchored_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
            )
            global_anchor_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                preserve_covered_targets=True,
            )
            balanced_anchor_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                preserve_covered_targets=True,
                balance_covered_target_anchors=True,
            )

        self.assertLess(
            covered_anchor_probability(unanchored_model),
            before_probability,
        )
        self.assertGreater(
            covered_anchor_probability(global_anchor_model),
            covered_anchor_probability(unanchored_model),
        )
        self.assertAlmostEqual(
            covered_anchor_probability(balanced_anchor_model),
            covered_anchor_probability(unanchored_model),
            places=12,
        )

    def test_branch_context_coverage_deficit_lifts_missing_target_probability(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )

        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=tokenizer.vocab_size,
                    context_size=8,
                    embedding_dim=4,
                    feedforward_dim=8,
                    seed=61,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        baseline_model = initialized_model()
        deficit_model = initialized_model()
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            baseline_model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        predicted_replay_targets = {
            predicted
            for _context, _target, predicted in replay_branches
            if predicted in replay_target_set
        }
        deficit_targets = replay_target_set - predicted_replay_targets
        self.assertTrue(deficit_targets)
        deficit_context, deficit_target, _predicted = next(
            branch
            for branch in replay_branches
            if branch[1] in deficit_targets
        )
        branch_batch = direct_answer_target_balanced_branch_diversity_batch(
            baseline_model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )

        def deficit_target_probability(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict(deficit_context)
            hard_candidates = [
                index
                for index in sorted(
                    range(len(probs)),
                    key=lambda item: probs[item],
                    reverse=True,
                )
                if index not in replay_target_set
            ][:5]
            candidate_ids = [*replay_targets, *hard_candidates]
            denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
            return probs[deficit_target] / denominator

        before_probability = deficit_target_probability(deficit_model)

        for _ in range(48):
            baseline_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
            )
            deficit_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                focus_uncovered_targets=True,
            )

        self.assertGreater(
            deficit_target_probability(deficit_model),
            before_probability,
        )
        self.assertGreater(
            deficit_target_probability(deficit_model),
            deficit_target_probability(baseline_model),
        )

    def test_branch_replay_plan_tracks_profile_deficits_independently(
        self,
    ) -> None:
        replay_branches = [
            ([0], 1, 1, "qa:place"),
            ([0], 2, 1, "qa:place"),
            ([0], 2, 2, "qa:color"),
        ]

        global_plan = branch_replay_plan(
            replay_branches,
            replay_branches,
            profile_aware_targets=False,
        )
        profile_plan = branch_replay_plan(
            replay_branches,
            replay_branches,
            profile_aware_targets=True,
        )

        self.assertEqual(
            global_plan["profiles"]["__all__"]["missing_target_ids"],
            [],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:place"]["missing_target_ids"],
            [2],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:color"]["missing_target_ids"],
            [],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:place"]["coverage_floor"],
            0.5,
        )
        with tempfile.TemporaryDirectory() as temp:
            plan_path = Path(temp) / "direct_answer_replay_plan.json"
            plan_path.write_text(
                json.dumps(profile_plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertTrue(loaded_plan["profile_aware_targets"])
        self.assertEqual(
            loaded_plan["profiles"]["qa:place"]["missing_target_count"],
            1,
        )

    def test_profiled_replay_records_preserve_sources_for_shared_targets(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        nine = AnswerExample(prompt="q: number?\na:", target=" nine.", source="qa:number")
        examples = [near, nine]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + nine.prompt
            + nine.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=63,
            )
        )

        records = direct_answer_profiled_replay_records(
            model,
            tokenizer,
            examples,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        target_ids_by_profile = {
            profile: target for _context, target, _predicted, profile in records
        }
        self.assertEqual(set(target_ids_by_profile), {"qa:place", "qa:number"})
        self.assertEqual(
            target_ids_by_profile["qa:place"],
            target_ids_by_profile["qa:number"],
        )

    def test_branch_replay_coverage_falls_back_to_sampled_branches(
        self,
    ) -> None:
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=3,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=4,
                seed=67,
            )
        )

        loss = model.train_step_with_branch_context_replay_coverage(
            [([0, 0, 0, 0], 1, 2)],
            [],
            learning_rate=0.01,
            negative_weight=0.0,
            positive_weight=0.0,
            replay_weight=1.0,
            hard_negative_count=1,
        )

        self.assertGreater(loss, 0.0)

    def test_branch_context_coverage_preserving_deficit_protects_represented_target(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )

        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=tokenizer.vocab_size,
                    context_size=8,
                    embedding_dim=4,
                    feedforward_dim=8,
                    seed=62,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        deficit_only_model = initialized_model()
        preserving_model = initialized_model()
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            deficit_only_model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        predicted_replay_targets = {
            predicted
            for _context, _target, predicted in replay_branches
            if predicted in replay_target_set
        }
        deficit_targets = replay_target_set - predicted_replay_targets
        self.assertTrue(deficit_targets)
        represented_context, _represented_target, represented_prediction = next(
            branch
            for branch in replay_branches
            if branch[2] in predicted_replay_targets
        )
        deficit_context, deficit_target, _predicted = next(
            branch
            for branch in replay_branches
            if branch[1] in deficit_targets
        )
        branch_batch = direct_answer_target_balanced_branch_diversity_batch(
            deficit_only_model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )

        def target_probability(
            scored_model: TinyTransformerLM,
            context: list[int],
            target: int,
        ) -> float:
            probs = scored_model.predict(context)
            hard_candidates = [
                index
                for index in sorted(
                    range(len(probs)),
                    key=lambda item: probs[item],
                    reverse=True,
                )
                if index not in replay_target_set
            ][:5]
            candidate_ids = [*replay_targets, *hard_candidates]
            denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
            return probs[target] / denominator

        before_deficit_probability = target_probability(
            preserving_model,
            deficit_context,
            deficit_target,
        )

        for _ in range(48):
            deficit_only_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                focus_uncovered_targets=True,
            )
            preserving_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                focus_uncovered_targets=True,
                preserve_predicted_target_coverage=True,
                balance_deficit_targets=True,
            )

        self.assertGreater(
            target_probability(preserving_model, deficit_context, deficit_target),
            before_deficit_probability,
        )
        self.assertGreater(
            target_probability(
                preserving_model,
                represented_context,
                represented_prediction,
            ),
            target_probability(
                deficit_only_model,
                represented_context,
                represented_prediction,
            ),
        )

    def test_branch_topk_softmax_lifts_target_within_hard_candidate_set(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=52,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def restricted_target_probability() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index != target
                ][:5]
                denominator = probs[target] + sum(
                    probs[candidate] for candidate in hard_candidates
                )
                total += probs[target] / denominator
            return total / len(batch)

        before_probability = restricted_target_probability()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_topk_softmax_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                candidate_weight=2.0,
                branch_position=1,
                batch_size=3,
                candidate_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertGreater(restricted_target_probability(), before_probability)

    def test_direct_answer_unlikelihood_penalizes_self_predicted_error(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" a.", source="qa:color")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=26,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        context, _target_id, predicted_id, _position = direct_answer_first_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )  # type: ignore[misc]
        before = model.predict(context)[predicted_id]
        rng = random.Random(5)

        for _ in range(24):
            train_direct_answer_first_error_unlikelihood(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        after = model.predict(context)[predicted_id]
        self.assertGreater(before, after)

    def test_direct_answer_rollout_error_uses_model_generated_prefix(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" bc.", source="qa:color")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=5,
                embedding_dim=3,
                feedforward_dim=5,
                seed=27,
            )
        )
        space_id = tokenizer.stoi[" "]
        b_id = tokenizer.stoi["b"]
        c_id = tokenizer.stoi["c"]
        model.bout[space_id].data = 4.0
        model.bout[b_id].data = 3.0
        model.bout[c_id].data = 2.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )

        repair = direct_answer_rollout_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair  # type: ignore[misc]
        before = model.predict(context)[predicted_id]
        rng = random.Random(6)
        for _ in range(16):
            train_direct_answer_rollout_unlikelihood(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )
        after = model.predict(context)[predicted_id]

        self.assertGreaterEqual(position, 1)
        self.assertNotEqual(target_id, predicted_id)
        self.assertGreater(before, after)

    def test_direct_answer_early_stop_penalizes_premature_terminator(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" a.", source="qa:color")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=28,
            )
        )
        terminator_id = tokenizer.stoi[ANSWER_TERMINATOR]
        model.bout[terminator_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )

        repair = direct_answer_early_stop_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair  # type: ignore[misc]
        before = model.predict(context)[predicted_id]
        rng = random.Random(7)
        for _ in range(24):
            train_direct_answer_early_stop_unlikelihood(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )
        after = model.predict(context)[predicted_id]

        self.assertEqual(tokenizer.itos[target_id], " ")
        self.assertEqual(tokenizer.itos[predicted_id], ANSWER_TERMINATOR)
        self.assertEqual(position, 0)
        self.assertGreater(before, after)

    def test_has_repeated_suffix_detects_repeated_bigram(self) -> None:
        self.assertTrue(has_repeated_suffix([1, 2, 1, 2]))
        self.assertTrue(has_repeated_suffix([3, 3]))
        self.assertFalse(has_repeated_suffix([1, 2, 1, 3]))

    def test_direct_answer_repeat_loop_penalizes_repeated_suffix(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=29,
            )
        )
        space_id = tokenizer.stoi[" "]
        model.bout[space_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )

        repair = direct_answer_repeat_loop_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair  # type: ignore[misc]
        before = model.predict(context)[predicted_id]
        rng = random.Random(8)
        for _ in range(24):
            train_direct_answer_repeat_loop_unlikelihood(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )
        after = model.predict(context)[predicted_id]

        self.assertEqual(tokenizer.itos[target_id], "n")
        self.assertEqual(tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertGreater(before, after)

    def test_direct_answer_balanced_repair_adds_positive_continuation(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=30,
            )
        )
        space_id = tokenizer.stoi[" "]
        model.bout[space_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        positive_lesson = [lesson[1]]
        positive_context, positive_target = positive_lesson[0]
        before_positive = model.nll(positive_context, positive_target)
        before_negative = model.predict(positive_context)[space_id]
        rng = random.Random(9)

        for _ in range(24):
            train_direct_answer_balanced_repair_unlikelihood(
                model,
                tokenizer,
                example,
                positive_lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        after_positive = model.nll(positive_context, positive_target)
        after_negative = model.predict(positive_context)[space_id]
        self.assertGreater(before_positive, after_positive)
        self.assertGreater(before_negative, after_negative)

    def test_direct_answer_generated_prefix_recovery_trains_after_bad_prefix(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=31,
            )
        )
        space_id = tokenizer.stoi[" "]
        model.bout[space_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        recovery = direct_answer_generated_prefix_recovery(
            model,
            tokenizer,
            example,
            recovery_steps=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(recovery)
        context, target_id, predicted_id, position, recovery_lesson = recovery  # type: ignore[misc]
        recovery_context, recovery_target = recovery_lesson[0]
        before_repair_negative = model.predict(context)[predicted_id]
        before_recovery = model.nll(recovery_context, recovery_target)
        rng = random.Random(10)

        for _ in range(24):
            train_direct_answer_generated_prefix_recovery_unlikelihood(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                recovery_steps=1,
                terminator=ANSWER_TERMINATOR,
            )

        after_repair_negative = model.predict(context)[predicted_id]
        after_recovery = model.nll(recovery_context, recovery_target)
        self.assertEqual(tokenizer.itos[target_id], "n")
        self.assertEqual(tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertEqual(tokenizer.itos[recovery_target], "n")
        self.assertGreater(before_repair_negative, after_repair_negative)
        self.assertGreater(before_recovery, after_recovery)

    def test_direct_answer_sequence_repair_collects_teacher_forced_errors(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=32,
            )
        )
        space_id = tokenizer.stoi[" "]
        model.bout[space_id].data = 5.0

        repairs = direct_answer_sequence_repair_errors(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )

        self.assertEqual(
            [
                position
                for _context, _target_id, _predicted_id, position in repairs
            ],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertTrue(
            all(
                predicted_id == space_id
                for _context, _target_id, predicted_id, _position in repairs
            )
        )

    def test_direct_answer_sequence_repair_reduces_sampled_errors(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=33,
            )
        )
        space_id = tokenizer.stoi[" "]
        model.bout[space_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        positive_lesson = [lesson[2]]
        positive_context, positive_target = positive_lesson[0]
        repairs = direct_answer_sequence_repair_errors(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(11)

        before_negative = sum(
            model.predict(context)[predicted_id]
            for context, _target_id, predicted_id, _position in repairs
        )
        before_positive = model.nll(positive_context, positive_target)
        for _ in range(40):
            train_direct_answer_sequence_repair_unlikelihood(
                model,
                tokenizer,
                example,
                positive_lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )
        after_negative = sum(
            model.predict(context)[predicted_id]
            for context, _target_id, predicted_id, _position in repairs
        )
        after_positive = model.nll(positive_context, positive_target)

        self.assertGreater(before_negative, after_negative)
        self.assertGreater(before_positive, after_positive)

    def test_direct_answer_loop_escape_pairs_loop_penalty_with_positive_path(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=34,
            )
        )
        space_id = tokenizer.stoi[" "]
        model.bout[space_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        positive_lesson = [lesson[1]]
        positive_context, positive_target = positive_lesson[0]
        repair = direct_answer_repeat_loop_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair  # type: ignore[misc]
        before_loop = model.predict(context)[predicted_id]
        before_positive = model.nll(positive_context, positive_target)
        rng = random.Random(12)

        for _ in range(32):
            train_direct_answer_loop_escape_unlikelihood(
                model,
                tokenizer,
                example,
                positive_lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        after_loop = model.predict(context)[predicted_id]
        after_positive = model.nll(positive_context, positive_target)
        self.assertEqual(tokenizer.itos[target_id], "n")
        self.assertEqual(tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertGreater(before_loop, after_loop)
        self.assertGreater(before_positive, after_positive)

    def test_direct_answer_branch_repair_targets_first_content_character(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=35,
            )
        )
        space_id = tokenizer.stoi[" "]
        model.bout[space_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        repair = direct_answer_branch_repair_error(
            model,
            tokenizer,
            example,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair  # type: ignore[misc]
        before = model.predict(context)[predicted_id]
        rng = random.Random(13)

        for _ in range(24):
            train_direct_answer_branch_repair_unlikelihood(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                terminator=ANSWER_TERMINATOR,
            )

        after = model.predict(context)[predicted_id]
        self.assertEqual(tokenizer.itos[target_id], "n")
        self.assertEqual(tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertGreater(before, after)

    def test_direct_answer_branch_span_samples_later_answer_positions(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        rng = random.Random(21)
        positions = {
            direct_answer_branch_span_position(
                tokenizer,
                example,
                rng,
                branch_position=1,
                branch_span=3,
                terminator=ANSWER_TERMINATOR,
            )
            for _ in range(24)
        }

        self.assertEqual(positions, {1, 2, 3})

    def test_direct_answer_branch_span_repair_targets_later_character(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" near.", source="qa:place")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=6,
                embedding_dim=3,
                feedforward_dim=5,
                seed=38,
            )
        )
        n_id = tokenizer.stoi["n"]
        model.bout[n_id].data = 5.0
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        repair = direct_answer_branch_span_repair_error(
            model,
            tokenizer,
            example,
            random.Random(1),
            branch_position=2,
            branch_span=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair  # type: ignore[misc]
        before = model.predict(context)[predicted_id]
        rng = random.Random(22)

        for _ in range(24):
            train_direct_answer_branch_span_repair_unlikelihood(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=2,
                branch_span=1,
                terminator=ANSWER_TERMINATOR,
            )

        after = model.predict(context)[predicted_id]
        self.assertEqual(tokenizer.itos[target_id], "e")
        self.assertEqual(tokenizer.itos[predicted_id], "n")
        self.assertEqual(position, 2)
        self.assertGreater(before, after)

    def test_direct_answer_branch_span_contrast_separates_later_branch(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + tree.prompt + tree.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=39,
            )
        )
        near_branch = direct_answer_branch_context(
            model,
            tokenizer,
            near,
            branch_position=2,
            terminator=ANSWER_TERMINATOR,
        )
        tree_branch = direct_answer_branch_context(
            model,
            tokenizer,
            tree,
            branch_position=2,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(near_branch)
        self.assertIsNotNone(tree_branch)
        near_context, near_target, _near_position = near_branch  # type: ignore[misc]
        tree_context, tree_target, _tree_position = tree_branch  # type: ignore[misc]
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        before = model.nll(near_context, near_target) + model.nll(tree_context, tree_target)
        rng = random.Random(23)

        for _ in range(64):
            train_direct_answer_branch_span_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                [tree],
                lesson,
                rng,
                learning_rate=0.05,
                negative_weight=1.0,
                positive_weight=1.0,
                contrast_weight=1.0,
                branch_position=2,
                branch_span=1,
                terminator=ANSWER_TERMINATOR,
            )

        after = model.nll(near_context, near_target) + model.nll(tree_context, tree_target)
        self.assertEqual(tokenizer.itos[near_target], "e")
        self.assertEqual(tokenizer.itos[tree_target], "r")
        self.assertGreater(before, after)

    def test_prompt_context_coverage_marks_truncated_semantic_prompt(self) -> None:
        records = [
            {
                "id": "color-teacher-tree",
                "prompt": "which color belongs to teacher tree\nanswer:",
                "target": " green.",
            }
        ]
        narrow = audit_prompt_context_coverage(records, context_size=32)
        wide = audit_prompt_context_coverage(records, context_size=64)

        self.assertEqual(narrow["semantic_records"], 1)
        self.assertEqual(narrow["missing"], 1)
        self.assertIn("intent:color", narrow["missing_records"][0]["missing_features"])
        self.assertEqual(wide["covered"], 1)
        self.assertEqual(wide["missing_records"], [])

    def test_direct_answer_branch_contrast_separates_prompt_branches(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=36,
            )
        )
        near_branch = direct_answer_branch_context(
            model,
            tokenizer,
            near,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        green_branch = direct_answer_branch_context(
            model,
            tokenizer,
            green,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(near_branch)
        self.assertIsNotNone(green_branch)
        near_context, near_target, _near_position = near_branch  # type: ignore[misc]
        green_context, green_target, _green_position = green_branch  # type: ignore[misc]
        near_lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        before_target = model.nll(near_context, near_target) + model.nll(
            green_context,
            green_target,
        )
        near_probs = model.predict(near_context)
        green_probs = model.predict(green_context)
        before_margin = (
            near_probs[near_target]
            - near_probs[green_target]
            + green_probs[green_target]
            - green_probs[near_target]
        )
        rng = random.Random(14)

        for _ in range(96):
            train_direct_answer_branch_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                [green],
                near_lesson,
                rng,
                learning_rate=0.05,
                negative_weight=1.0,
                contrast_weight=1.0,
                branch_position=1,
                terminator=ANSWER_TERMINATOR,
            )

        after_target = model.nll(near_context, near_target) + model.nll(
            green_context,
            green_target,
        )
        near_probs = model.predict(near_context)
        green_probs = model.predict(green_context)
        after_margin = (
            near_probs[near_target]
            - near_probs[green_target]
            + green_probs[green_target]
            - green_probs[near_target]
        )
        self.assertEqual(tokenizer.itos[near_target], "n")
        self.assertEqual(tokenizer.itos[green_target], "g")
        self.assertGreater(before_target, after_target)
        self.assertGreater(after_margin, before_margin)

    def test_direct_answer_hard_branch_contrast_selects_confused_branch(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=37,
            )
        )
        model.bout[tokenizer.stoi["t"]].data = 4.0
        model.bout[tokenizer.stoi["g"]].data = 1.0
        contrast = direct_answer_hard_branch_contrast(
            model,
            tokenizer,
            near,
            [green, tree],
            random.Random(15),
            branch_position=1,
            hard_negative_count=0,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(contrast)
        context, target_id, _contrast_context, contrast_target = contrast  # type: ignore[misc]
        before_wrong = model.predict(context)[contrast_target]
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )

        for _ in range(24):
            train_direct_answer_hard_branch_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                [green, tree],
                lesson,
                random.Random(16),
                learning_rate=0.05,
                negative_weight=1.0,
                positive_weight=1.0,
                contrast_weight=1.0,
                branch_position=1,
                hard_negative_count=0,
                terminator=ANSWER_TERMINATOR,
            )

        after_wrong = model.predict(context)[contrast_target]
        self.assertEqual(tokenizer.itos[target_id], "n")
        self.assertEqual(tokenizer.itos[contrast_target], "t")
        self.assertGreater(before_wrong, after_wrong)

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
                "closed_world_lm.answer_model corpus-derived AnswerExample lessons"
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
