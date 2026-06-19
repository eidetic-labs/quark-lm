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
    TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS,
    build_torch_accumulation_replay_control_probe,
    build_torch_accumulation_replay_plan,
    build_torch_training_state,
    torch_runtime_status,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchAccumulationReplayControlTests(unittest.TestCase):
    def test_probe_records_microstep_backward_control(self) -> None:
        fixture, state, torch, runtime = _ready_inputs()
        plan = build_torch_accumulation_replay_plan(fixture=fixture)

        probe = build_torch_accumulation_replay_control_probe(
            fixture=fixture,
            state=state,
            torch=torch,
            runtime=runtime,
            replay_plan=plan,
        )

        self.assertEqual(
            probe["status"],
            TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS,
        )
        self.assertEqual(probe["planned_microstep_count"], 2)
        self.assertEqual(probe["executed_microstep_count"], 2)
        self.assertEqual(probe["backward_pass_count"], 2)
        self.assertEqual(probe["optimizer_updates_applied"], 0)
        self.assertFalse(probe["accumulated_gradient_parity_proven"])
        self.assertFalse(probe["final_update_parity_proven"])
        self.assertEqual(
            probe["microsteps"][0]["clip_report"]["status"],
            "gradient_clip_applied",
        )
        self.assertFalse(probe["microsteps"][1]["optimizer_step_applied"])
        json.dumps(probe)

    def test_probe_waits_for_runtime_state(self) -> None:
        fixture, _state, _torch, runtime = _ready_inputs()

        probe = build_torch_accumulation_replay_control_probe(
            fixture=fixture,
            state=None,
            torch=None,
            runtime=runtime,
            replay_plan=build_torch_accumulation_replay_plan(fixture=fixture),
        )

        self.assertEqual(probe["status"], "not_run")


def _ready_inputs() -> tuple[dict, dict, object, dict]:
    fixture = _scalar_training_fixture()
    importer = fake_torch_importer(
        training_runtime=True,
        gradient_runtime=True,
    )
    torch = importer("torch")
    runtime = torch_runtime_status(importer=importer)
    state = build_torch_training_state(
        fixture=fixture,
        torch=torch,
        runtime=runtime,
    )
    return fixture, state, torch, runtime


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
        ),
        learning_rate=0.02,
        steps=2,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
