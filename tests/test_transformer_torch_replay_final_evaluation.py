from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from support.fake_torch import fake_torch_importer
from transformer_model import OptimizationConfig
from transformer_torch_backend import (
    TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
    TORCH_REPLAY_FINAL_EVAL_NOT_RUN_STATUS,
    build_torch_replay_buffer_comparison,
    build_torch_replay_final_evaluation,
    build_torch_replay_update_comparison,
    build_torch_training_state,
    torch_runtime_status,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchReplayFinalEvaluationTests(unittest.TestCase):
    def test_final_evaluation_matches_after_replayed_update(self) -> None:
        fixture, torch, runtime = _ready_runtime()
        control = _scalar_matching_replay_control(fixture)
        buffer_comparison = build_torch_replay_buffer_comparison(
            fixture=fixture,
            replay_control_probe=control,
        )
        update_comparison = build_torch_replay_update_comparison(
            fixture=fixture,
            state=_training_state(fixture, torch, runtime),
            torch=torch,
            runtime=runtime,
            replay_control_probe=control,
            buffer_comparison=buffer_comparison,
        )

        final_eval = build_torch_replay_final_evaluation(
            fixture=fixture,
            state=_training_state(fixture, torch, runtime),
            torch=torch,
            runtime=runtime,
            replay_control_probe=control,
            buffer_comparison=buffer_comparison,
            update_comparison=update_comparison,
        )

        self.assertEqual(final_eval["status"], TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS)
        self.assertTrue(final_eval["passed"])
        self.assertTrue(final_eval["final_logit_parity_proven"])
        self.assertTrue(final_eval["final_loss_parity_proven"])
        self.assertFalse(final_eval["checkpoint_parity_proven"])
        self.assertLessEqual(final_eval["loss_abs_diff"], 1e-9)
        self.assertLessEqual(final_eval["max_logit_abs_diff"], 1e-9)
        json.dumps(final_eval)

    def test_final_evaluation_waits_for_update_parity(self) -> None:
        fixture, torch, runtime = _ready_runtime()

        final_eval = build_torch_replay_final_evaluation(
            fixture=fixture,
            state=_training_state(fixture, torch, runtime),
            torch=torch,
            runtime=runtime,
            replay_control_probe=_scalar_matching_replay_control(fixture),
            buffer_comparison={"passed": True},
            update_comparison={"status": "replay_update_signature_mismatch"},
        )

        self.assertEqual(final_eval["status"], TORCH_REPLAY_FINAL_EVAL_NOT_RUN_STATUS)
        self.assertFalse(final_eval["final_logit_parity_proven"])
        self.assertFalse(final_eval["final_loss_parity_proven"])
        self.assertIn("update", final_eval["reason"])


def _ready_runtime() -> tuple[dict, object, dict]:
    fixture = _scalar_training_fixture()
    importer = fake_torch_importer(
        training_runtime=True,
        gradient_runtime=True,
    )
    torch = importer("torch")
    runtime = torch_runtime_status(importer=importer)
    return fixture, torch, runtime


def _training_state(fixture: dict, torch: object, runtime: dict) -> dict:
    return build_torch_training_state(
        fixture=fixture,
        torch=torch,
        runtime=runtime,
    )


def _scalar_training_fixture() -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="tiny-training-scalar",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=OptimizationConfig(
            optimizer="adamw",
            gradient_accumulation_steps=2,
            warmup_steps=2,
            decay_steps=2,
            min_learning_rate=0.001,
        ),
        learning_rate=0.02,
        steps=2,
        corpus_hash="corpus-hash",
    )


def _scalar_matching_replay_control(fixture: dict) -> dict:
    return {
        "status": "accumulation_replay_control_recorded",
        "case_id": fixture["training_case"]["case_id"],
        "microsteps": [
            {
                "step": record["step"],
                "gradient_snapshot": _snapshot(
                    record["optimizer_gradient_evidence"]["clipped_gradient"][
                        "values"
                    ]
                ),
            }
            for record in fixture["training_case"]["step_records"]
        ],
    }


def _snapshot(values: list[float]) -> dict:
    return {
        "parameters": [
            {
                "name": "flat-gradient",
                "values": list(values),
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
