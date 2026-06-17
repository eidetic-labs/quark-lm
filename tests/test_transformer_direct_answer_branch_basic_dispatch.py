from __future__ import annotations

import random
import unittest
from argparse import Namespace
from unittest.mock import patch

from transformer_direct_answer_branch_basic_dispatch import (
    BASIC_BRANCH_DIRECT_ANSWER_MODES,
    train_direct_answer_branch_basic_mode_step,
)


class TransformerDirectAnswerBranchBasicDispatchTests(unittest.TestCase):
    def test_branch_repair_mode_routes_to_repair_objective(self) -> None:
        args = _args("branch-repair-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_basic_trainers.train_direct_answer_branch_repair_unlikelihood",
            return_value=1.25,
        ) as repair:
            loss = _train(args)

        self.assertEqual(loss, 1.25)
        repair.assert_called_once()
        self.assertEqual(repair.call_args.args[7], args.direct_answer_positive_weight)
        self.assertEqual(repair.call_args.args[8], args.direct_answer_branch_position)

    def test_periodic_branch_repair_uses_first_error_fallback_off_interval(self) -> None:
        args = _args("periodic-branch-repair-unlikelihood", interval=3)

        with (
            patch(
                "transformer_direct_answer_branch_basic_trainers.train_direct_answer_first_error_unlikelihood",
                return_value=2.5,
            ) as first_error,
            patch(
                "transformer_direct_answer_branch_basic_trainers.train_direct_answer_branch_repair_unlikelihood",
                return_value=9.0,
            ) as repair,
        ):
            loss = _train(args, direct_step=2)

        self.assertEqual(loss, 2.5)
        first_error.assert_called_once()
        repair.assert_not_called()

    def test_periodic_branch_diversity_uses_repair_fallback_off_interval(self) -> None:
        args = _args("periodic-branch-diversity-unlikelihood", interval=5)

        with (
            patch(
                "transformer_direct_answer_branch_basic_trainers.train_direct_answer_branch_diversity_unlikelihood",
                return_value=8.0,
            ) as diversity,
            patch(
                "transformer_direct_answer_branch_basic_trainers.train_direct_answer_branch_repair_unlikelihood",
                return_value=3.75,
            ) as repair,
        ):
            loss = _train(args, direct_step=3)

        self.assertEqual(loss, 3.75)
        repair.assert_called_once()
        diversity.assert_not_called()

    def test_target_margin_mode_routes_to_margin_objective(self) -> None:
        args = _args("branch-target-margin-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_basic_trainers.train_direct_answer_branch_target_margin_unlikelihood",
            return_value=4.0,
        ) as margin:
            loss = _train(args)

        self.assertEqual(loss, 4.0)
        margin.assert_called_once()
        self.assertEqual(margin.call_args.args[9], args.direct_answer_contrast_weight)

    def test_mode_set_contains_basic_branch_modes(self) -> None:
        self.assertIn("branch-collapse-unlikelihood", BASIC_BRANCH_DIRECT_ANSWER_MODES)
        self.assertIn("periodic-branch-target-softmax-unlikelihood", BASIC_BRANCH_DIRECT_ANSWER_MODES)
        self.assertIn("branch-hidden-projection-margin-unlikelihood", BASIC_BRANCH_DIRECT_ANSWER_MODES)


def _args(mode: str, interval: int = 2) -> Namespace:
    return Namespace(
        direct_answer_mode=mode,
        direct_answer_learning_rate=0.1,
        direct_answer_negative_weight=0.2,
        direct_answer_positive_weight=0.3,
        direct_answer_contrast_weight=0.4,
        direct_answer_branch_position=1,
        direct_answer_hard_negatives=2,
        direct_answer_branch_batch_size=3,
        direct_answer_rollout_interval=interval,
    )


def _train(args: Namespace, direct_step: int = 2) -> float:
    return train_direct_answer_branch_basic_mode_step(
        args=args,
        model=object(),
        tokenizer=object(),
        example=object(),
        lesson=[],
        branch_examples=[],
        rng=random.Random(7),
        direct_step=direct_step,
        terminator="<END>",
        params=[],
    )


if __name__ == "__main__":
    unittest.main()
