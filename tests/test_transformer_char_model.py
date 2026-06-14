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
    direct_answer_rollout_error,
    direct_answer_early_stop_error,
    direct_answer_repeat_loop_error,
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
    train_direct_answer_lesson,
    train_answer_char,
    train_answer_mixed_step,
    direct_answer_lesson,
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
                ]
            )
            self.assertEqual(args.direct_answer_mode, mode)
            self.assertEqual(args.direct_answer_rollout_interval, 4)
            self.assertEqual(args.direct_answer_positive_weight, 1.5)

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
