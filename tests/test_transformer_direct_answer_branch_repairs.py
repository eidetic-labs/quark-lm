from __future__ import annotations

import random
import unittest

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    direct_answer_branch_context,
    direct_answer_branch_repair_error,
    direct_answer_branch_span_position,
    direct_answer_branch_span_repair_error,
    direct_answer_lesson,
    train_direct_answer_branch_repair_unlikelihood,
    train_direct_answer_branch_span_contrast_unlikelihood,
    train_direct_answer_branch_span_repair_unlikelihood,
)
from support.direct_answer_repairs import direct_answer_repair_fixture


class TransformerDirectAnswerBranchRepairsTest(unittest.TestCase):
    def test_direct_answer_branch_repair_targets_first_content_character(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=35,
        )
        space_id = fixture.tokenizer.stoi[" "]
        fixture.model.bout[space_id].data = 5.0
        repair = direct_answer_branch_repair_error(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair
        before = fixture.model.predict(context)[predicted_id]
        rng = random.Random(13)

        for _ in range(24):
            train_direct_answer_branch_repair_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                fixture.lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertEqual(fixture.tokenizer.itos[target_id], "n")
        self.assertEqual(fixture.tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertGreater(before, fixture.model.predict(context)[predicted_id])

    def test_direct_answer_branch_span_samples_later_answer_positions(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=35,
        )
        rng = random.Random(21)
        positions = {
            direct_answer_branch_span_position(
                fixture.tokenizer,
                fixture.example,
                rng,
                branch_position=1,
                branch_span=3,
                terminator=ANSWER_TERMINATOR,
            )
            for _ in range(24)
        }

        self.assertEqual(positions, {1, 2, 3})

    def test_direct_answer_branch_span_repair_targets_later_character(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=38,
        )
        n_id = fixture.tokenizer.stoi["n"]
        fixture.model.bout[n_id].data = 5.0
        repair = direct_answer_branch_span_repair_error(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            random.Random(1),
            branch_position=2,
            branch_span=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair
        before = fixture.model.predict(context)[predicted_id]
        rng = random.Random(22)

        for _ in range(24):
            train_direct_answer_branch_span_repair_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                fixture.lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=2,
                branch_span=1,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertEqual(fixture.tokenizer.itos[target_id], "e")
        self.assertEqual(fixture.tokenizer.itos[predicted_id], "n")
        self.assertEqual(position, 2)
        self.assertGreater(before, fixture.model.predict(context)[predicted_id])

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
        near_context, near_target, _near_position = near_branch
        tree_context, tree_target, _tree_position = tree_branch
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        before = model.nll(near_context, near_target) + model.nll(
            tree_context,
            tree_target,
        )
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

        after = model.nll(near_context, near_target) + model.nll(
            tree_context,
            tree_target,
        )
        self.assertEqual(tokenizer.itos[near_target], "e")
        self.assertEqual(tokenizer.itos[tree_target], "r")
        self.assertGreater(before, after)


if __name__ == "__main__":
    unittest.main()
