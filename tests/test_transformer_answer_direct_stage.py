from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformer_answer_direct_stage import (
    restore_stage_state_and_rebind_recorder,
    run_transformer_direct_answer_stage,
)


class TransformerAnswerDirectStageTest(unittest.TestCase):
    def test_restore_rebinds_snapshot_recorder_to_stage_state(self) -> None:
        stage_state = _FakeStageState()
        recorder = SimpleNamespace(
            model=lambda: "old-model",
            tokenizer=lambda: "old-tokenizer",
        )

        restored = restore_stage_state_and_rebind_recorder(
            stage_state,
            recorder,
            {"model": "restored-model"},
            {"optimizer": "restored-optimizer"},
        )

        self.assertEqual(stage_state.restore_calls, 1)
        self.assertEqual(recorder.model(), "restored-model")
        self.assertEqual(recorder.tokenizer(), "restored-tokenizer")
        self.assertEqual(stage_state.optimizer, "restored-optimizer")
        self.assertEqual(
            restored,
            (
                "restored-model",
                "restored-tokenizer",
                "restored-optimizer",
                ["restored-param"],
            ),
        )

    def test_stage_restore_callback_returns_refreshed_state(self) -> None:
        captured: dict[str, object] = {}

        def run_loop(**kwargs: object) -> SimpleNamespace:
            restore = kwargs["restore_direct_update_state"]
            captured["restored"] = restore({"model": True}, {"optimizer": True})
            return SimpleNamespace(
                last_snapshot={"step": 0},
                last_snapshot_step=0,
                routing_repair_batch_evidence=None,
            )

        with (
            patch(
                "transformer_answer_direct_stage.prepare_direct_answer_run_setup",
                return_value=_direct_setup(),
            ),
            patch(
                "transformer_answer_direct_stage.initialize_direct_answer_phase",
                return_value=_direct_runtime(),
            ),
            patch(
                "transformer_answer_direct_stage.build_baseline_anchored_prompt_updater",
                return_value=SimpleNamespace(train=lambda *args: 0.0),
            ),
            patch(
                "transformer_answer_direct_stage.build_stabilization_context",
                return_value=object(),
            ),
            patch(
                "transformer_answer_direct_stage.build_stabilization_trainer",
                return_value=lambda *args: (0.0, False),
            ),
            patch(
                "transformer_answer_direct_stage.build_adaptive_baseline_floor_trainer",
                return_value=lambda *args: 0.0,
            ),
            patch(
                "transformer_answer_direct_stage.run_direct_answer_training_loop",
                side_effect=run_loop,
            ),
            patch(
                "transformer_answer_direct_stage.complete_direct_answer_phase",
                return_value=_direct_phase(),
            ),
            patch(
                "transformer_answer_direct_stage_state.restore_direct_answer_update_state",
                return_value=(
                    "restored-model",
                    "restored-tokenizer",
                    "restored-optimizer",
                    ["restored-param"],
                ),
            ),
        ):
            run_transformer_direct_answer_stage(
                args=SimpleNamespace(
                    direct_answer_train_top_layer_only=False,
                    direct_answer_freeze_output_bias=False,
                ),
                model_class=object,
                setup=SimpleNamespace(
                    examples=[],
                    eval_records=[],
                    generation_config=object(),
                    context_coverage={},
                    training_plan_path="plan.json",
                ),
                model="model",
                tokenizer="tokenizer",
                optimizer="optimizer",
                training_plan={},
                last_snapshot={},
                snapshot=object(),
            )

        self.assertEqual(
            captured["restored"],
            (
                "restored-model",
                "restored-tokenizer",
                "restored-optimizer",
                ["restored-param"],
            ),
        )


class _FakeStageState:
    def __init__(self) -> None:
        self.model = "old-model"
        self.tokenizer = "old-tokenizer"
        self.optimizer = "old-optimizer"
        self.params = ["old-param"]
        self.restore_calls = 0

    def restore(
        self,
        model_payload: dict[str, str],
        optimizer_payload: dict[str, str],
    ) -> None:
        self.restore_calls += 1
        self.model = model_payload["model"]
        self.tokenizer = "restored-tokenizer"
        self.optimizer = optimizer_payload["optimizer"]
        self.params = ["restored-param"]


def _direct_setup() -> SimpleNamespace:
    return SimpleNamespace(
        training_plan={},
        direct_training_pool=[],
        direct_lessons={},
        direct_answer_terminator="\n",
        direct_rng=object(),
        direct_answer_baseline_floor_update_gate_active=False,
        direct_answer_baseline_floor_adaptive_updates_active=False,
    )


def _direct_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        params=[],
        update_guard={},
        baseline={},
        last_snapshot={},
        last_snapshot_step=0,
        snapshot_recorder=SimpleNamespace(),
        best_snapshot=object(),
        training_cursor=object(),
        steps_to_run=0,
        training_skipped=False,
        skip_reason=None,
        branch_context_gate={},
    )


def _direct_phase() -> SimpleNamespace:
    return SimpleNamespace(
        model="phase-model",
        tokenizer="phase-tokenizer",
        optimizer="phase-optimizer",
        last_snapshot={},
        post_direct_candidate_snapshot=None,
        post_direct_candidate_snapshot_skipped=True,
        metrics={},
    )


if __name__ == "__main__":
    unittest.main()
