from __future__ import annotations

import copy
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
    TORCH_OPTIMIZER_STEP_READY_STATUS,
    build_torch_optimizer_step_probe,
    build_torch_training_backward_probe,
    build_torch_training_state,
    torch_runtime_status,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchOptimizerStepProbeTests(unittest.TestCase):
    def test_probe_waits_for_backward_gradients(self) -> None:
        fixture = _scalar_training_fixture()
        importer = fake_torch_importer(training_runtime=True)
        runtime = torch_runtime_status(importer=importer)
        state = build_torch_training_state(
            fixture=fixture,
            torch=importer("torch"),
            runtime=runtime,
        )
        backward_probe = build_torch_training_backward_probe(
            fixture=fixture,
            state=state,
            torch=importer("torch"),
            runtime=runtime,
        )

        probe = build_torch_optimizer_step_probe(
            fixture=fixture,
            state=state,
            backward_probe=backward_probe,
        )

        self.assertEqual(probe["status"], "not_run")
        self.assertEqual(probe["reason"], "pytorch gradients are not available")

    def test_probe_reports_ready_when_gradients_satisfy_contract(self) -> None:
        fixture = _scalar_training_fixture()
        importer = fake_torch_importer(
            training_runtime=True,
            gradient_runtime=True,
        )
        runtime = torch_runtime_status(importer=importer)
        state = build_torch_training_state(
            fixture=fixture,
            torch=importer("torch"),
            runtime=runtime,
        )
        backward_probe = build_torch_training_backward_probe(
            fixture=fixture,
            state=state,
            torch=importer("torch"),
            runtime=runtime,
        )

        probe = build_torch_optimizer_step_probe(
            fixture=fixture,
            state=state,
            backward_probe=backward_probe,
        )

        summary = probe["gradient_summary"]
        self.assertEqual(probe["status"], TORCH_OPTIMIZER_STEP_READY_STATUS)
        self.assertEqual(
            probe["expected_step_count"],
            fixture["training_case"]["steps"],
        )
        self.assertEqual(probe["expected_update_count"], 1)
        self.assertEqual(summary["missing_gradient_tensor_count"], 0)
        self.assertEqual(summary["shape_mismatch_count"], 0)
        self.assertTrue(summary["parameter_order_matches"])
        self.assertTrue(summary["parameter_index_coverage_matches"])
        self.assertEqual(
            summary["gradient_parameter_count"],
            fixture["optimizer_step_contract"]["parameter_count"],
        )
        json.dumps(probe)

    def test_probe_rejects_gradient_shape_mismatch(self) -> None:
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
        state["parameters"][0]["tensor"].grad = torch.tensor([0.0])

        probe = build_torch_optimizer_step_probe(
            fixture=fixture,
            state=state,
            backward_probe=backward_probe,
        )

        self.assertEqual(probe["status"], "gradient_shape_mismatch")
        self.assertEqual(
            probe["gradient_summary"]["shape_mismatch_parameters"],
            [state["parameters"][0]["name"]],
        )

    def test_probe_rejects_invalid_contract(self) -> None:
        fixture = copy.deepcopy(_scalar_training_fixture())
        fixture["optimizer_step_contract"]["schema_version"] = -1
        importer = fake_torch_importer(
            training_runtime=True,
            gradient_runtime=True,
        )
        runtime = torch_runtime_status(importer=importer)
        state = build_torch_training_state(
            fixture=fixture,
            torch=importer("torch"),
            runtime=runtime,
        )
        backward_probe = build_torch_training_backward_probe(
            fixture=fixture,
            state=state,
            torch=importer("torch"),
            runtime=runtime,
        )

        probe = build_torch_optimizer_step_probe(
            fixture=fixture,
            state=state,
            backward_probe=backward_probe,
        )

        self.assertEqual(probe["status"], "contract_invalid")
        self.assertIn("schema_version", probe["reason"])


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
