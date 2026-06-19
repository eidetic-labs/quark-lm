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
    TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS,
    build_torch_adamw_expected_update,
    build_torch_training_backward_probe,
    build_torch_training_state,
    snapshot_torch_parameters,
    torch_runtime_status,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchAdamWExpectedUpdateTests(unittest.TestCase):
    def test_expected_update_uses_current_clipped_gradients(self) -> None:
        fixture, state, torch = _ready_state()
        first = state["parameters"][0]
        first["tensor"].grad = torch.tensor(_filled(first["shape"], 5.0))
        before = snapshot_torch_parameters(state)

        expected = build_torch_adamw_expected_update(
            state=state,
            parameters_before=before,
            contract=fixture["optimizer_step_contract"],
        )

        self.assertEqual(expected["status"], TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS)
        self.assertEqual(
            expected["expected_signature"]["count"],
            fixture["parameter_manifest"]["parameter_count"],
        )
        self.assertGreater(expected["gradient_signature"]["sum"], 0.0)
        self.assertNotEqual(expected["expected_signature"], before["signature"])
        json.dumps(expected)

    def test_expected_update_rejects_multiple_updates(self) -> None:
        fixture, state, _torch = _ready_state(
            optimizer_config=OptimizationConfig(
                optimizer="adamw",
                gradient_accumulation_steps=1,
            ),
            steps=2,
        )

        expected = build_torch_adamw_expected_update(
            state=state,
            parameters_before=snapshot_torch_parameters(state),
            contract=fixture["optimizer_step_contract"],
        )

        self.assertEqual(expected["status"], "not_built")
        self.assertIn("exactly one update", expected["reason"])


def _ready_state(
    *,
    optimizer_config: OptimizationConfig | None = None,
    steps: int = 1,
) -> tuple[dict, dict, object]:
    fixture = _scalar_training_fixture(
        optimizer_config=optimizer_config
        or OptimizationConfig(optimizer="adamw"),
        steps=steps,
    )
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
    build_torch_training_backward_probe(
        fixture=fixture,
        state=state,
        torch=torch,
        runtime=runtime,
    )
    return fixture, state, torch


def _filled(shape: list[int], value: float) -> object:
    if not shape:
        return value
    return [_filled(shape[1:], value) for _index in range(shape[0])]


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
