from __future__ import annotations

import random
import unittest

from transformer_char_model_test_support import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
    audit_prompt_context_coverage,
    direct_answer_branch_context,
    direct_answer_branch_repair_error,
    direct_answer_branch_span_position,
    direct_answer_branch_span_repair_error,
    direct_answer_early_stop_error,
    direct_answer_first_error,
    direct_answer_generated_prefix_recovery,
    direct_answer_hard_branch_contrast,
    direct_answer_lesson,
    direct_answer_repeat_loop_error,
    direct_answer_rollout_error,
    direct_answer_sequence_repair_errors,
    direct_answer_target_balanced_branch_diversity_batch,
    has_repeated_suffix,
    train_direct_answer_balanced_repair_unlikelihood,
    train_direct_answer_branch_contrast_unlikelihood,
    train_direct_answer_branch_repair_unlikelihood,
    train_direct_answer_branch_span_contrast_unlikelihood,
    train_direct_answer_branch_span_repair_unlikelihood,
    train_direct_answer_branch_topk_softmax_unlikelihood,
    train_direct_answer_early_stop_unlikelihood,
    train_direct_answer_first_error_unlikelihood,
    train_direct_answer_generated_prefix_recovery_unlikelihood,
    train_direct_answer_hard_branch_contrast_unlikelihood,
    train_direct_answer_loop_escape_unlikelihood,
    train_direct_answer_repeat_loop_unlikelihood,
    train_direct_answer_rollout_unlikelihood,
    train_direct_answer_sequence_repair_unlikelihood,
)


class TransformerDirectAnswerRepairsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
