from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_lesson,
    train_direct_answer_profile_balanced_branch_rank_collapse_unlikelihood,
    train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood,
    train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood,
    train_direct_answer_profile_balanced_retention_rank_margin_unlikelihood,
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
        calls: list[
            tuple[
                list[tuple[list[int], int, int]],
                list[tuple[list[int], int, int, str]],
                dict[str, object],
            ]
        ] = []
        represented_target = fixture.tokenizer.stoi[fixture.near.target[1]]

        def predict_represented_target(_context: list[int]) -> list[float]:
            probs = [0.0 for _token in fixture.tokenizer.tokens]
            probs[represented_target] = 1.0
            return probs

        def train_step(
            branches: list[tuple[list[int], int, int]],
            retention_anchors: list[tuple[list[int], int, int, str]],
            *_args: object,
            **kwargs: object,
        ) -> float:
            calls.append((branches, retention_anchors, kwargs))
            return 4.5

        fixture.model.predict = predict_represented_target
        fixture.model.train_step_with_branch_retention_topk_softmax = train_step

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
        topk_branches, retention_anchors, kwargs = calls[0]
        self.assertEqual(len(topk_branches), 2)
        self.assertGreater(len(retention_anchors), 0)
        self.assertEqual(kwargs["representation_weight"], 2.0)
        self.assertTrue(all(len(branch) == 3 for branch in topk_branches))
        self.assertTrue(all(len(branch) == 4 for branch in retention_anchors))

    def test_profile_balanced_rank_collapse_uses_profile_balanced_branches(
        self,
    ) -> None:
        fixture = branch_training_fixture(seed=53)
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        calls: list[list[tuple[list[int], int, int]]] = []
        losses = [7.0, 4.0]

        def train_step(
            branches: list[tuple[list[int], int, int]],
            *_args: object,
            **_kwargs: object,
        ) -> float:
            calls.append(branches)
            return losses[len(calls) - 1]

        fixture.model.train_step_with_branch_rank_collapse_margin = train_step

        loss = train_direct_answer_profile_balanced_branch_rank_collapse_unlikelihood(
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

        self.assertEqual(loss, 5.0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(len(call) for call in calls), [1, 2])
        self.assertTrue(
            all(len(branch) == 3 for call in calls for branch in call)
        )

    def test_retention_rank_margin_runs_single_combined_step(self) -> None:
        fixture = branch_training_fixture(seed=52)
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        calls: list[
            tuple[
                list[tuple[list[int], int, int]],
                list[tuple[list[int], int, int, str]],
            ]
        ] = []

        def combined_step(
            branches: list[tuple[list[int], int, int]],
            retention_anchors: list[tuple[list[int], int, int, str]],
            *_args: object,
            **_kwargs: object,
        ) -> float:
            calls.append((branches, retention_anchors))
            return 4.0

        fixture.model.train_step_with_branch_retention_rank_margin = combined_step

        loss = train_direct_answer_profile_balanced_retention_rank_margin_unlikelihood(
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

        self.assertEqual(loss, 4.0)
        self.assertEqual(len(calls), 1)
        rank_branches, retention_anchors = calls[0]
        self.assertTrue(all(len(branch) == 3 for branch in rank_branches))
        self.assertTrue(all(len(branch) == 4 for branch in retention_anchors))


if __name__ == "__main__":
    unittest.main()
