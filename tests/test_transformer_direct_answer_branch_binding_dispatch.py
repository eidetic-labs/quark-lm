from __future__ import annotations

import random
import unittest
from argparse import Namespace
from unittest.mock import patch

from transformer_direct_answer_branch_binding_dispatch import (
    BRANCH_BINDING_DIRECT_ANSWER_MODES,
    train_direct_answer_branch_binding_mode_step,
)


class TransformerDirectAnswerBranchBindingDispatchTests(unittest.TestCase):
    def test_representation_mode_routes_to_unbalanced_objective(self) -> None:
        args = _args("branch-representation-contrast-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_binding_dispatch."
            "train_direct_answer_branch_representation_contrast_unlikelihood",
            return_value=1.0,
        ) as representation:
            loss = _train(args)

        self.assertEqual(loss, 1.0)
        representation.assert_called_once()
        self.assertFalse(representation.call_args.kwargs["balance_targets"])

    def test_balanced_bidirectional_mode_sets_balanced_targets(self) -> None:
        args = _args("branch-balanced-bidirectional-binding-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_binding_dispatch."
            "train_direct_answer_branch_bidirectional_binding_unlikelihood",
            return_value=2.0,
        ) as bidirectional:
            loss = _train(args)

        self.assertEqual(loss, 2.0)
        bidirectional.assert_called_once()
        self.assertTrue(bidirectional.call_args.kwargs["balance_targets"])

    def test_coverage_mode_passes_hard_negative_count(self) -> None:
        args = _args("branch-coverage-binding-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_binding_dispatch."
            "train_direct_answer_branch_coverage_binding_unlikelihood",
            return_value=3.0,
        ) as coverage:
            loss = _train(args)

        self.assertEqual(loss, 3.0)
        coverage.assert_called_once()
        self.assertEqual(coverage.call_args.args[12], args.direct_answer_hard_negatives)
        self.assertFalse(coverage.call_args.kwargs["balance_targets"])

    def test_balanced_target_replay_sets_balanced_targets(self) -> None:
        args = _args("branch-balanced-target-replay-coverage-unlikelihood")

        with patch(
            "transformer_direct_answer_branch_binding_dispatch."
            "train_direct_answer_branch_target_replay_coverage_unlikelihood",
            return_value=4.0,
        ) as target_replay:
            loss = _train(args)

        self.assertEqual(loss, 4.0)
        target_replay.assert_called_once()
        self.assertEqual(target_replay.call_args.args[12], args.direct_answer_hard_negatives)
        self.assertTrue(target_replay.call_args.kwargs["balance_targets"])

    def test_mode_set_contains_binding_and_target_replay_modes(self) -> None:
        self.assertIn("branch-output-binding-unlikelihood", BRANCH_BINDING_DIRECT_ANSWER_MODES)
        self.assertIn("branch-balanced-target-diversity-unlikelihood", BRANCH_BINDING_DIRECT_ANSWER_MODES)
        self.assertIn("branch-target-replay-coverage-unlikelihood", BRANCH_BINDING_DIRECT_ANSWER_MODES)


def _args(mode: str) -> Namespace:
    return Namespace(
        direct_answer_mode=mode,
        direct_answer_learning_rate=0.1,
        direct_answer_negative_weight=0.2,
        direct_answer_positive_weight=0.3,
        direct_answer_contrast_weight=0.4,
        direct_answer_branch_position=1,
        direct_answer_hard_negatives=2,
        direct_answer_branch_batch_size=3,
    )


def _train(args: Namespace) -> float:
    return train_direct_answer_branch_binding_mode_step(
        args=args,
        model=object(),
        tokenizer=object(),
        example=object(),
        lesson=[],
        branch_examples=[],
        rng=random.Random(11),
        terminator="<END>",
        params=[],
    )


if __name__ == "__main__":
    unittest.main()
