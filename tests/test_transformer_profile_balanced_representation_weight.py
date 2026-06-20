from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_lesson,
    train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood,
)


class TransformerProfileBalancedRepresentationWeightTest(unittest.TestCase):
    def test_profile_balanced_topk_can_opt_into_representation_pressure(self) -> None:
        fixture = branch_training_fixture(seed=51)
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        calls: list[dict[str, object]] = []

        def train_step(
            _branches: list[tuple[list[int], int, int]],
            _retention_anchors: list[tuple[list[int], int, int, str]],
            *_args: object,
            **kwargs: object,
        ) -> float:
            calls.append(kwargs)
            return 4.5

        fixture.model.train_step_with_branch_retention_topk_softmax = train_step

        train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            lesson,
            random.Random(15),
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=1.0,
            candidate_weight=2.0,
            branch_position=1,
            batch_size=2,
            candidate_count=5,
            terminator=ANSWER_TERMINATOR,
            representation_weight=0.25,
        )

        self.assertEqual(calls[0]["representation_weight"], 0.25)


if __name__ == "__main__":
    unittest.main()
