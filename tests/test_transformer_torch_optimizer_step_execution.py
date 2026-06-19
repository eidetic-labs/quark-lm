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
    TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS,
    TORCH_OPTIMIZER_STEP_READY_STATUS,
    build_torch_optimizer_step_execution_probe,
    build_torch_optimizer_step_probe,
    build_torch_training_backward_probe,
    build_torch_training_state,
    torch_runtime_status,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchOptimizerStepExecutionTests(unittest.TestCase):
    def test_execution_waits_for_optimizer_readiness(self) -> None:
        probe = build_torch_optimizer_step_execution_probe(
            fixture=_scalar_training_fixture(),
            state=None,
            optimizer_step_probe={"status": "not_run"},
            torch=None,
        )

        self.assertEqual(probe["status"], "not_run")
        self.assertEqual(probe["reason"], "pytorch training runtime is not ready")

    def test_execution_matches_scalar_step_control_contract(self) -> None:
        fixture, state, optimizer_probe, torch = _ready_optimizer_inputs()

        execution = build_torch_optimizer_step_execution_probe(
            fixture=fixture,
            state=state,
            optimizer_step_probe=optimizer_probe,
            torch=torch,
        )

        expected = fixture["optimizer_step_contract"]
        self.assertEqual(
            execution["status"],
            TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS,
        )
        self.assertTrue(execution["step_records_match_contract"])
        self.assertTrue(execution["final_state_matches_contract"])
        self.assertEqual(
            execution["optimizer_state"],
            expected["expected_final_optimizer_state"],
        )
        self.assertEqual(execution["applied_update_count"], 1)
        self.assertFalse(execution["step_records"][0]["optimizer_step_called"])
        self.assertTrue(execution["step_records"][1]["optimizer_step_called"])
        json.dumps(execution)

    def test_execution_reports_unavailable_optimizer(self) -> None:
        fixture, state, optimizer_probe, torch = _ready_optimizer_inputs()
        torch.optim.AdamW = None

        execution = build_torch_optimizer_step_execution_probe(
            fixture=fixture,
            state=state,
            optimizer_step_probe=optimizer_probe,
            torch=torch,
        )

        self.assertEqual(execution["status"], "optimizer_unavailable")
        self.assertIn("AdamW", execution["reason"])


def _ready_optimizer_inputs() -> tuple[dict, dict, dict, object]:
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
    backward_probe = build_torch_training_backward_probe(
        fixture=fixture,
        state=state,
        torch=torch,
        runtime=runtime,
    )
    optimizer_probe = build_torch_optimizer_step_probe(
        fixture=fixture,
        state=state,
        backward_probe=backward_probe,
    )
    assert optimizer_probe["status"] == TORCH_OPTIMIZER_STEP_READY_STATUS
    return fixture, state, optimizer_probe, torch


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


if __name__ == "__main__":
    unittest.main()
