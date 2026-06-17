from __future__ import annotations

import random
import unittest

from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_repeat_loop_error,
    has_repeated_suffix,
    train_direct_answer_balanced_repair_unlikelihood,
    train_direct_answer_loop_escape_unlikelihood,
    train_direct_answer_repeat_loop_unlikelihood,
)
from support.direct_answer_repairs import direct_answer_repair_fixture


class TransformerDirectAnswerLoopRepairsTest(unittest.TestCase):
    def test_has_repeated_suffix_detects_repeated_bigram(self) -> None:
        self.assertTrue(has_repeated_suffix([1, 2, 1, 2]))
        self.assertTrue(has_repeated_suffix([3, 3]))
        self.assertFalse(has_repeated_suffix([1, 2, 1, 3]))

    def test_direct_answer_repeat_loop_penalizes_repeated_suffix(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=29,
        )
        space_id = fixture.tokenizer.stoi[" "]
        fixture.model.bout[space_id].data = 5.0

        repair = direct_answer_repeat_loop_error(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair
        before = fixture.model.predict(context)[predicted_id]
        rng = random.Random(8)
        for _ in range(24):
            train_direct_answer_repeat_loop_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                fixture.lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertEqual(fixture.tokenizer.itos[target_id], "n")
        self.assertEqual(fixture.tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertGreater(before, fixture.model.predict(context)[predicted_id])

    def test_direct_answer_balanced_repair_adds_positive_continuation(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=30,
        )
        space_id = fixture.tokenizer.stoi[" "]
        fixture.model.bout[space_id].data = 5.0
        positive_lesson = [fixture.lesson[1]]
        positive_context, positive_target = positive_lesson[0]
        before_positive = fixture.model.nll(positive_context, positive_target)
        before_negative = fixture.model.predict(positive_context)[space_id]
        rng = random.Random(9)

        for _ in range(24):
            train_direct_answer_balanced_repair_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                positive_lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(before_positive, fixture.model.nll(positive_context, positive_target))
        self.assertGreater(before_negative, fixture.model.predict(positive_context)[space_id])

    def test_direct_answer_loop_escape_pairs_loop_penalty_with_positive_path(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=34,
        )
        space_id = fixture.tokenizer.stoi[" "]
        fixture.model.bout[space_id].data = 5.0
        positive_lesson = [fixture.lesson[1]]
        positive_context, positive_target = positive_lesson[0]
        repair = direct_answer_repeat_loop_error(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair
        before_loop = fixture.model.predict(context)[predicted_id]
        before_positive = fixture.model.nll(positive_context, positive_target)
        rng = random.Random(12)

        for _ in range(32):
            train_direct_answer_loop_escape_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                positive_lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertEqual(fixture.tokenizer.itos[target_id], "n")
        self.assertEqual(fixture.tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertGreater(before_loop, fixture.model.predict(context)[predicted_id])
        self.assertGreater(before_positive, fixture.model.nll(positive_context, positive_target))


if __name__ == "__main__":
    unittest.main()
