from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_lesson,
    train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood,
    train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood,
)


class TransformerProfileBalancedRankObjectiveTest(unittest.TestCase):
    def test_profile_balanced_rank_margin_uses_profile_balanced_branches(self) -> None:
        fixture = branch_training_fixture(seed=50)
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        calls: list[list[tuple[list[int], int, int]]] = []

        def train_step(
            branches: list[tuple[list[int], int, int]],
            *_args: object,
            **_kwargs: object,
        ) -> float:
            calls.append(branches)
            return 3.5

        fixture.model.train_step_with_branch_rank_margin = train_step

        loss = train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            lesson,
            random.Random(15),
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=1.0,
            margin_weight=2.0,
            branch_position=1,
            batch_size=2,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(loss, 3.5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertTrue(all(len(branch) == 3 for branch in calls[0]))

    def test_profile_balanced_topk_softmax_uses_profile_balanced_branches(self) -> None:
        fixture = branch_training_fixture(seed=51)
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        calls: list[list[tuple[list[int], int, int]]] = []

        def train_step(
            branches: list[tuple[list[int], int, int]],
            *_args: object,
            **_kwargs: object,
        ) -> float:
            calls.append(branches)
            return 4.5

        fixture.model.train_step_with_branch_topk_softmax = train_step

        loss = train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood(
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
        )

        self.assertEqual(loss, 4.5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertTrue(all(len(branch) == 3 for branch in calls[0]))


if __name__ == "__main__":
    unittest.main()
