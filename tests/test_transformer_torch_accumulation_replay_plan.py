from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from transformer_model import OptimizationConfig
from transformer_torch_backend import (
    TORCH_ACCUMULATION_REPLAY_PENDING_STATUS,
    build_torch_accumulation_replay_plan,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchAccumulationReplayPlanTests(unittest.TestCase):
    def test_plan_records_clipped_buffer_microsteps(self) -> None:
        fixture = _scalar_training_fixture(
            optimizer_config=OptimizationConfig(
                optimizer="adamw",
                gradient_accumulation_steps=2,
            ),
            steps=2,
        )

        plan = build_torch_accumulation_replay_plan(fixture=fixture)

        self.assertEqual(plan["status"], TORCH_ACCUMULATION_REPLAY_PENDING_STATUS)
        self.assertEqual(plan["microstep_count"], 2)
        self.assertTrue(plan["requires_microstep_clipping"])
        self.assertTrue(plan["requires_clipped_gradient_buffer"])
        self.assertFalse(plan["native_loss_scaling_sufficient"])
        self.assertEqual(plan["microsteps"][0]["loss_scale"], 1.0)
        self.assertEqual(
            plan["microsteps"][0]["buffer_action"],
            "clip_then_buffer_gradient",
        )
        self.assertFalse(plan["microsteps"][0]["optimizer_step_after_microstep"])
        self.assertTrue(plan["microsteps"][1]["optimizer_step_after_microstep"])
        self.assertFalse(plan["execution_status"]["replayed_backward_passes"])
        self.assertFalse(plan["accumulated_gradient_parity_proven"])
        json.dumps(plan)

    def test_plan_records_loss_scaling_when_buffer_is_not_required(self) -> None:
        fixture = _scalar_training_fixture(
            optimizer_config=OptimizationConfig(
                optimizer="adamw",
                gradient_accumulation_steps=4,
                gradient_clip=0.0,
            ),
            steps=4,
        )

        plan = build_torch_accumulation_replay_plan(fixture=fixture)

        self.assertFalse(plan["requires_clipped_gradient_buffer"])
        self.assertTrue(plan["native_loss_scaling_sufficient"])
        self.assertEqual(plan["microsteps"][0]["loss_scale"], 0.25)
        self.assertEqual(
            plan["microsteps"][0]["buffer_action"],
            "accumulate_scaled_gradient",
        )
        self.assertTrue(plan["microsteps"][3]["optimizer_step_after_microstep"])


def _scalar_training_fixture(
    *,
    optimizer_config: OptimizationConfig,
    steps: int,
) -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="tiny-training-scalar",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=optimizer_config,
        learning_rate=0.02,
        steps=steps,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
