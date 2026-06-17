from __future__ import annotations

import random
import unittest

from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_early_stop_error,
    direct_answer_first_error,
    direct_answer_rollout_error,
    train_direct_answer_early_stop_unlikelihood,
    train_direct_answer_first_error_unlikelihood,
    train_direct_answer_rollout_unlikelihood,
)
from support.direct_answer_repairs import direct_answer_repair_fixture


class TransformerDirectAnswerRepairsTest(unittest.TestCase):
    def test_direct_answer_unlikelihood_penalizes_self_predicted_error(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" a.",
            source="qa:color",
            seed=26,
            context_size=4,
        )
        wrong_id = fixture.tokenizer.stoi["."]
        fixture.model.bout[wrong_id].data = 5.0
        repair = direct_answer_first_error(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, _target_id, predicted_id, _position = repair
        before = fixture.model.predict(context)[predicted_id]
        rng = random.Random(5)

        for _ in range(24):
            train_direct_answer_first_error_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                fixture.lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(before, fixture.model.predict(context)[predicted_id])

    def test_direct_answer_rollout_error_uses_model_generated_prefix(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" bc.",
            source="qa:color",
            seed=27,
            context_size=5,
        )
        space_id = fixture.tokenizer.stoi[" "]
        b_id = fixture.tokenizer.stoi["b"]
        c_id = fixture.tokenizer.stoi["c"]
        fixture.model.bout[space_id].data = 4.0
        fixture.model.bout[b_id].data = 3.0
        fixture.model.bout[c_id].data = 2.0

        repair = direct_answer_rollout_error(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair
        before = fixture.model.predict(context)[predicted_id]
        rng = random.Random(6)
        for _ in range(16):
            train_direct_answer_rollout_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                fixture.lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreaterEqual(position, 1)
        self.assertNotEqual(target_id, predicted_id)
        self.assertGreater(before, fixture.model.predict(context)[predicted_id])

    def test_direct_answer_early_stop_penalizes_premature_terminator(self) -> None:
        fixture = direct_answer_repair_fixture(
            target=" a.",
            source="qa:color",
            seed=28,
            context_size=4,
        )
        terminator_id = fixture.tokenizer.stoi[ANSWER_TERMINATOR]
        fixture.model.bout[terminator_id].data = 5.0

        repair = direct_answer_early_stop_error(
            fixture.model,
            fixture.tokenizer,
            fixture.example,
            ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(repair)
        context, target_id, predicted_id, position = repair
        before = fixture.model.predict(context)[predicted_id]
        rng = random.Random(7)
        for _ in range(24):
            train_direct_answer_early_stop_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.example,
                fixture.lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertEqual(fixture.tokenizer.itos[target_id], " ")
        self.assertEqual(fixture.tokenizer.itos[predicted_id], ANSWER_TERMINATOR)
        self.assertEqual(position, 0)
        self.assertGreater(before, fixture.model.predict(context)[predicted_id])


if __name__ == "__main__":
    unittest.main()
