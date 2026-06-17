from __future__ import annotations

import random
import unittest
from argparse import Namespace
from unittest.mock import patch

from transformer_direct_answer_context_replay_dispatch import (
    CONTEXT_REPLAY_DIRECT_ANSWER_MODES,
    train_direct_answer_context_replay_mode_step,
)


class TransformerDirectAnswerContextReplayDispatchTests(unittest.TestCase):
    def test_plain_context_replay_mode_uses_default_flags(self) -> None:
        args = _args("branch-context-replay-coverage-unlikelihood")

        with patch(
            "transformer_direct_answer_context_replay_dispatch."
            "train_direct_answer_branch_context_replay_coverage_unlikelihood",
            return_value=1.0,
        ) as context_replay:
            loss = _train(args)

        self.assertEqual(loss, 1.0)
        context_replay.assert_called_once()
        self.assertEqual(
            context_replay.call_args.args[12],
            args.direct_answer_hard_negatives,
        )
        self.assertFalse(context_replay.call_args.kwargs["balance_targets"])
        self.assertFalse(context_replay.call_args.kwargs["focus_uncovered_targets"])

    def test_target_balanced_anchor_mode_sets_anchor_flags(self) -> None:
        args = _args("branch-context-target-balanced-anchor-unlikelihood")

        with patch(
            "transformer_direct_answer_context_replay_dispatch."
            "train_direct_answer_branch_context_replay_coverage_unlikelihood",
            return_value=2.0,
        ) as context_replay:
            loss = _train(args)

        self.assertEqual(loss, 2.0)
        self.assertFalse(context_replay.call_args.kwargs["balance_targets"])
        self.assertTrue(context_replay.call_args.kwargs["preserve_covered_targets"])
        self.assertTrue(context_replay.call_args.kwargs["balance_covered_target_anchors"])

    def test_balanced_profile_prompt_ownership_mode_sets_profile_flags(self) -> None:
        args = _args(
            "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood"
        )

        with patch(
            "transformer_direct_answer_context_replay_dispatch."
            "train_direct_answer_branch_context_replay_coverage_unlikelihood",
            return_value=3.0,
        ) as context_replay:
            loss = _train(args)

        self.assertEqual(loss, 3.0)
        kwargs = context_replay.call_args.kwargs
        self.assertTrue(kwargs["balance_targets"])
        self.assertTrue(kwargs["focus_uncovered_targets"])
        self.assertTrue(kwargs["preserve_predicted_target_coverage"])
        self.assertTrue(kwargs["balance_deficit_targets"])
        self.assertTrue(kwargs["profile_aware_targets"])
        self.assertTrue(kwargs["balance_profile_target_shares"])
        self.assertTrue(kwargs["enforce_prompt_target_margins"])

    def test_mode_set_contains_context_replay_modes(self) -> None:
        self.assertEqual(len(CONTEXT_REPLAY_DIRECT_ANSWER_MODES), 14)
        self.assertIn(
            "branch-balanced-context-replay-coverage-unlikelihood",
            CONTEXT_REPLAY_DIRECT_ANSWER_MODES,
        )
        self.assertIn(
            "branch-context-profile-coverage-preserving-deficit-unlikelihood",
            CONTEXT_REPLAY_DIRECT_ANSWER_MODES,
        )


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
    return train_direct_answer_context_replay_mode_step(
        args=args,
        model=object(),
        tokenizer=object(),
        example=object(),
        lesson=[],
        branch_examples=[],
        rng=random.Random(13),
        terminator="<END>",
        params=[],
    )


if __name__ == "__main__":
    unittest.main()
