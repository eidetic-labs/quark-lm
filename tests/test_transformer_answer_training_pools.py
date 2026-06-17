from __future__ import annotations

import unittest

from support.core import AnswerExample
from support.direct_answer import (
    transformer_answer_generator_training_pool,
    transformer_direct_answer_training_pool,
)


class TransformerAnswerTrainingPoolsTest(unittest.TestCase):
    def test_transformer_generator_pool_prioritizes_long_operational_lessons(
        self,
    ) -> None:
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


if __name__ == "__main__":
    unittest.main()
