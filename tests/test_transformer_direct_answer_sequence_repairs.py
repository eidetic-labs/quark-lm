from __future__ import annotations

import random
import unittest

from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_generated_prefix_recovery,
    direct_answer_sequence_repair_errors,
    train_direct_answer_generated_prefix_recovery_unlikelihood,
    train_direct_answer_sequence_repair_unlikelihood,
)
from support.direct_answer_repairs import direct_answer_repair_fixture


class TransformerDirectAnswerSequenceRepairsTest(unittest.TestCase):
    def test_direct_answer_generated_prefix_recovery_trains_after_bad_prefix(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=31,
        )
        space_id = fixture.tokenizer.stoi[" "]
        fixture.model.bout[space_id].data = 5.0
        recovery = direct_answer_generated_prefix_recovery(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            recovery_steps=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(recovery)
        context, target_id, predicted_id, position, recovery_lesson = recovery
        recovery_context, recovery_target = recovery_lesson[0]
        before_repair_negative = fixture.model.predict(context)[predicted_id]
        before_recovery = fixture.model.nll(recovery_context, recovery_target)
        rng = random.Random(10)

        for _ in range(24):
            train_direct_answer_generated_prefix_recovery_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                fixture.lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                recovery_steps=1,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertEqual(fixture.tokenizer.itos[target_id], "n")
        self.assertEqual(fixture.tokenizer.itos[predicted_id], " ")
        self.assertEqual(position, 1)
        self.assertEqual(fixture.tokenizer.itos[recovery_target], "n")
        self.assertGreater(
            before_repair_negative,
            fixture.model.predict(context)[predicted_id],
        )
        self.assertGreater(
            before_recovery,
            fixture.model.nll(recovery_context, recovery_target),
        )

    def test_direct_answer_sequence_repair_collects_teacher_forced_errors(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=32,
        )
        space_id = fixture.tokenizer.stoi[" "]
        fixture.model.bout[space_id].data = 5.0

        repairs = direct_answer_sequence_repair_errors(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
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
        fixture = direct_answer_repair_fixture(
            target=" near.",
            source="qa:place",
            seed=33,
        )
        space_id = fixture.tokenizer.stoi[" "]
        fixture.model.bout[space_id].data = 5.0
        positive_lesson = [fixture.lesson[2]]
        positive_context, positive_target = positive_lesson[0]
        repairs = direct_answer_sequence_repair_errors(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(11)

        before_negative = sum(
            fixture.model.predict(context)[predicted_id]
            for context, _target_id, predicted_id, _position in repairs
        )
        before_positive = fixture.model.nll(positive_context, positive_target)
        for _ in range(40):
            train_direct_answer_sequence_repair_unlikelihood(
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
        after_negative = sum(
            fixture.model.predict(context)[predicted_id]
            for context, _target_id, predicted_id, _position in repairs
        )

        self.assertGreater(before_negative, after_negative)
        self.assertGreater(
            before_positive,
            fixture.model.nll(positive_context, positive_target),
        )


if __name__ == "__main__":
    unittest.main()
