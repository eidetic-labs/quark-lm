from __future__ import annotations

import random
import unittest
from argparse import Namespace
from unittest.mock import Mock, patch

from transformer_direct_answer_mode_dispatch import train_direct_answer_mode_step


BASELINE_MODE = (
    "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)


class TransformerDirectAnswerModeDispatchTests(unittest.TestCase):
    def test_error_repair_result_short_circuits_family_dispatch(self) -> None:
        args = _args("first-error-unlikelihood")

        with (
            patch(
                "transformer_direct_answer_mode_dispatch."
                "train_direct_answer_error_repair_mode_step",
                return_value=1.5,
            ) as error_dispatch,
            patch(
                "transformer_direct_answer_mode_dispatch."
                "train_direct_answer_branch_basic_mode_step",
                return_value=9.0,
            ) as basic_dispatch,
        ):
            result = _train(args)

        self.assertEqual(result.loss, 1.5)
        self.assertFalse(result.update_guard_applied)
        error_dispatch.assert_called_once()
        basic_dispatch.assert_not_called()

    def test_basic_branch_mode_routes_to_basic_family(self) -> None:
        args = _args("branch-repair-unlikelihood")

        with (
            patch(
                "transformer_direct_answer_mode_dispatch."
                "train_direct_answer_error_repair_mode_step",
                return_value=None,
            ),
            patch(
                "transformer_direct_answer_mode_dispatch."
                "train_direct_answer_branch_basic_mode_step",
                return_value=2.0,
            ) as basic_dispatch,
        ):
            result = _train(args)

        self.assertEqual(result.loss, 2.0)
        basic_dispatch.assert_called_once()

    def test_baseline_mode_uses_adaptive_callback_when_payloads_exist(self) -> None:
        args = _args(BASELINE_MODE)
        adaptive = Mock(return_value=3.25)
        baseline = Mock(return_value=8.0)

        with patch(
            "transformer_direct_answer_mode_dispatch."
            "train_direct_answer_error_repair_mode_step",
            return_value=None,
        ):
            result = _train(
                args,
                adaptive_active=True,
                pre_update_model_payload={"model": True},
                pre_update_optimizer_payload={"optimizer": True},
                pre_update_rng_state=("rng",),
                adaptive_callback=adaptive,
                baseline_callback=baseline,
            )

        self.assertEqual(result.loss, 3.25)
        self.assertTrue(result.update_guard_applied)
        adaptive.assert_called_once_with(
            2,
            {"model": True},
            {"optimizer": True},
            ("rng",),
        )
        baseline.assert_not_called()

    def test_baseline_mode_uses_prompt_callback_without_adaptive_payloads(self) -> None:
        args = _args(BASELINE_MODE)
        adaptive = Mock(return_value=7.0)
        baseline = Mock(return_value=4.5)

        with patch(
            "transformer_direct_answer_mode_dispatch."
            "train_direct_answer_error_repair_mode_step",
            return_value=None,
        ):
            result = _train(
                args,
                adaptive_active=True,
                adaptive_callback=adaptive,
                baseline_callback=baseline,
            )

        self.assertEqual(result.loss, 4.5)
        self.assertFalse(result.update_guard_applied)
        adaptive.assert_not_called()
        baseline.assert_called_once()

    def test_unknown_mode_falls_back_to_lesson_training(self) -> None:
        args = _args("lesson")

        with (
            patch(
                "transformer_direct_answer_mode_dispatch."
                "train_direct_answer_error_repair_mode_step",
                return_value=None,
            ),
            patch(
                "transformer_direct_answer_mode_dispatch."
                "train_direct_answer_lesson",
                return_value=5.0,
            ) as lesson_train,
        ):
            result = _train(args)

        self.assertEqual(result.loss, 5.0)
        lesson_train.assert_called_once()


def _args(mode: str) -> Namespace:
    return Namespace(
        direct_answer_mode=mode,
        direct_answer_learning_rate=0.1,
    )


def _train(
    args: Namespace,
    *,
    adaptive_active: bool = False,
    pre_update_model_payload: dict[str, object] | None = None,
    pre_update_optimizer_payload: dict[str, object] | None = None,
    pre_update_rng_state: object | None = None,
    adaptive_callback: Mock | None = None,
    baseline_callback: Mock | None = None,
):
    return train_direct_answer_mode_step(
        args=args,
        model=object(),
        tokenizer=object(),
        example=object(),
        lesson=[],
        branch_examples=[],
        rng=random.Random(19),
        direct_step=2,
        terminator="<END>",
        params=[],
        baseline_floor_adaptive_updates_active=adaptive_active,
        pre_update_model_payload=pre_update_model_payload,
        pre_update_optimizer_payload=pre_update_optimizer_payload,
        pre_update_rng_state=pre_update_rng_state,
        train_adaptive_baseline_floor_update=adaptive_callback or Mock(),
        train_baseline_anchored_prompt=baseline_callback or Mock(),
    )


if __name__ == "__main__":
    unittest.main()
