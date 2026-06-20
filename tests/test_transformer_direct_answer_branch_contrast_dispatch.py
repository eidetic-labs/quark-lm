from __future__ import annotations

import random
import unittest
from argparse import Namespace
from unittest.mock import patch

from transformer_direct_answer_branch_contrast_dispatch import (
    BRANCH_CONTRAST_DIRECT_ANSWER_MODES,
    train_direct_answer_branch_contrast_mode_step,
)


class TransformerDirectAnswerBranchContrastDispatchTests(unittest.TestCase):
    def test_balanced_rank_margin_sets_balanced_targets(self) -> None:
        args = _args("branch-balanced-rank-margin-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_contrast_adapters."
            "train_direct_answer_branch_rank_margin_unlikelihood",
            return_value=1.0,
        ) as rank_margin:
            loss = _train(args)

        self.assertEqual(loss, 1.0)
        rank_margin.assert_called_once()
        self.assertEqual(
            rank_margin.call_args.args[12],
            args.direct_answer_hard_negatives,
        )
        self.assertTrue(rank_margin.call_args.kwargs["balance_targets"])

    def test_profile_balanced_rank_margin_routes_to_profile_balanced_objective(self) -> None:
        args = _args("branch-profile-balanced-rank-margin-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_profile_balanced_adapters."
            "train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood",
            return_value=1.5,
        ) as profile_balanced_rank:
            loss = _train(args)

        self.assertEqual(loss, 1.5)
        profile_balanced_rank.assert_called_once()
        self.assertEqual(
            profile_balanced_rank.call_args.args[12],
            args.direct_answer_hard_negatives,
        )

    def test_profile_balanced_topk_routes_to_profile_balanced_objective(self) -> None:
        args = _args("branch-profile-balanced-topk-softmax-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_profile_balanced_adapters."
            "train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood",
            return_value=1.75,
        ) as profile_balanced_topk:
            loss = _train(args)

        self.assertEqual(loss, 1.75)
        profile_balanced_topk.assert_called_once()
        self.assertEqual(
            profile_balanced_topk.call_args.args[12],
            args.direct_answer_hard_negatives,
        )

    def test_periodic_span_repair_uses_first_error_fallback_off_interval(self) -> None:
        args = _args("periodic-branch-span-repair-unlikelihood", interval=3)

        with (
            patch(
                "transformer_direct_answer_branch_contrast_adapters."
                "train_direct_answer_first_error_unlikelihood",
                return_value=2.0,
            ) as first_error,
            patch(
                "transformer_direct_answer_branch_contrast_adapters."
                "train_direct_answer_branch_span_repair_unlikelihood",
                return_value=9.0,
            ) as span_repair,
        ):
            loss = _train(args, direct_step=2)

        self.assertEqual(loss, 2.0)
        first_error.assert_called_once()
        span_repair.assert_not_called()

    def test_periodic_repair_contrast_uses_repair_fallback_off_interval(self) -> None:
        args = _args("periodic-branch-repair-contrast-unlikelihood", interval=4)

        with (
            patch(
                "transformer_direct_answer_branch_contrast_adapters."
                "train_direct_answer_branch_repair_unlikelihood",
                return_value=3.0,
            ) as repair,
            patch(
                "transformer_direct_answer_branch_contrast_adapters."
                "train_direct_answer_branch_contrast_unlikelihood",
                return_value=8.0,
            ) as contrast,
        ):
            loss = _train(args, direct_step=3)

        self.assertEqual(loss, 3.0)
        repair.assert_called_once()
        contrast.assert_not_called()

    def test_periodic_hard_repair_contrast_runs_hard_contrast_on_interval(self) -> None:
        args = _args("periodic-hard-branch-repair-contrast-unlikelihood", interval=2)

        with (
            patch(
                "transformer_direct_answer_branch_contrast_adapters."
                "train_direct_answer_hard_branch_contrast_unlikelihood",
                return_value=4.0,
            ) as hard_contrast,
            patch(
                "transformer_direct_answer_branch_contrast_adapters."
                "train_direct_answer_branch_repair_unlikelihood",
                return_value=7.0,
            ) as repair,
        ):
            loss = _train(args, direct_step=2)

        self.assertEqual(loss, 4.0)
        hard_contrast.assert_called_once()
        self.assertEqual(
            hard_contrast.call_args.args[11],
            args.direct_answer_hard_negatives,
        )
        repair.assert_not_called()

    def test_mode_set_contains_rank_and_contrast_modes(self) -> None:
        self.assertIn(
            "branch-profile-balanced-rank-margin-unlikelihood",
            BRANCH_CONTRAST_DIRECT_ANSWER_MODES,
        )
        self.assertIn(
            "branch-profile-balanced-topk-softmax-unlikelihood",
            BRANCH_CONTRAST_DIRECT_ANSWER_MODES,
        )
        self.assertIn(
            "branch-balanced-topk-softmax-unlikelihood",
            BRANCH_CONTRAST_DIRECT_ANSWER_MODES,
        )
        self.assertIn(
            "periodic-branch-representation-contrast-unlikelihood",
            BRANCH_CONTRAST_DIRECT_ANSWER_MODES,
        )
        self.assertIn(
            "periodic-branch-span-repair-contrast-unlikelihood",
            BRANCH_CONTRAST_DIRECT_ANSWER_MODES,
        )


def _args(mode: str, interval: int = 2) -> Namespace:
    return Namespace(
        direct_answer_mode=mode,
        direct_answer_learning_rate=0.1,
        direct_answer_negative_weight=0.2,
        direct_answer_positive_weight=0.3,
        direct_answer_contrast_weight=0.4,
        direct_answer_branch_position=1,
        direct_answer_branch_batch_size=3,
        direct_answer_hard_negatives=2,
        direct_answer_branch_span=5,
        direct_answer_rollout_interval=interval,
    )


def _train(args: Namespace, direct_step: int = 2) -> float:
    return train_direct_answer_branch_contrast_mode_step(
        args=args,
        model=object(),
        tokenizer=object(),
        example=object(),
        lesson=[],
        branch_examples=[],
        rng=random.Random(17),
        direct_step=direct_step,
        terminator="<END>",
        params=[],
    )


if __name__ == "__main__":
    unittest.main()
