from __future__ import annotations

import sys
import tempfile
import unittest
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.neural_char_model import context_before, continuation_nll
from closed_world_lm.tokenizer import CharTokenizer
from closed_world_lm.answer_model import AnswerExample
from closed_world_lm.transformer_char_model import (
    AnswerCandidateSelector,
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
    direct_answer_branch_span_position,
    direct_answer_branch_span_repair_error,
    direct_answer_branch_batch,
    direct_answer_dominant_branch_prediction,
    direct_answer_hard_branch_contrast,
    audit_prompt_context_coverage,
    evaluate_direct_answer_records,
    evaluate_answer_generator_records,
    evaluate_answer_records,
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
        self.assertEqual(profile["target_tokens"][0], {"value": " ", "count": 1})
        self.assertEqual(profile["predicted_tokens"][0], {"value": ".", "count": 1})
        self.assertEqual(profile["confusions"][0], {"value": "' '->'.'", "count": 1})
        self.assertEqual(profile["failed_records"][0]["id"], "color")
        self.assertEqual(profile["failed_records"][0]["target_token"], " ")
        self.assertEqual(profile["failed_records"][0]["predicted_token"], ".")

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
                    "--direct-answer-train-top-layer-only",
                    "--skip-post-direct-snapshot",
                    "--direct-answer-sequence-interval",
                    "6",
                    "--num-layers",
                    "2",
                    "--use-layer-norm",
                    "--layer-norm-epsilon",
                    "0.0001",
                    "--use-context-mean",
                    "--use-context-projection",
                    "--use-prompt-attention-summary",
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
            self.assertTrue(args.direct_answer_train_top_layer_only)
            self.assertTrue(args.skip_post_direct_snapshot)
            self.assertEqual(args.num_layers, 2)
            self.assertTrue(args.use_layer_norm)
            self.assertEqual(args.layer_norm_epsilon, 0.0001)
            self.assertTrue(args.use_context_mean)
            self.assertTrue(args.use_context_projection)
            self.assertTrue(args.use_prompt_attention_summary)
            self.assertEqual(args.direct_answer_sequence_interval, 6)

    def test_parse_train_args_accepts_context_mean(self) -> None:
        args = parse_args(
            [
                "train",
                "--use-context-mean",
                "--use-context-projection",
                "--use-prompt-attention-summary",
            ]
        )

        self.assertTrue(args.use_context_mean)
        self.assertTrue(args.use_context_projection)
        self.assertTrue(args.use_prompt_attention_summary)

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
