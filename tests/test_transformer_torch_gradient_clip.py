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
    TORCH_GRADIENT_CLIP_APPLIED_STATUS,
    apply_torch_gradient_value_clip,
    build_torch_training_state,
    torch_runtime_status,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchGradientClipTests(unittest.TestCase):
    def test_value_clip_mutates_oversized_gradients(self) -> None:
        fixture, state, torch = _state_with_first_gradient(9.0)

        report = apply_torch_gradient_value_clip(
            torch=torch,
            state=state,
            clip_value=fixture["optimizer_config"]["gradient_clip"],
        )

        self.assertEqual(report["status"], TORCH_GRADIENT_CLIP_APPLIED_STATUS)
        self.assertTrue(report["applied"])
        self.assertEqual(report["max_abs_before"], 9.0)
        self.assertEqual(report["max_abs_after"], 5.0)
        self.assertGreater(report["changed_scalar_count"], 0)
        self.assertEqual(state["parameters"][0]["tensor"].grad.tolist()[0][0], 5.0)
        json.dumps(report)

    def test_value_clip_reports_missing_runtime_capability(self) -> None:
        _fixture, state, torch = _state_with_first_gradient(9.0)
        torch.nn.utils.clip_grad_value_ = None

        report = apply_torch_gradient_value_clip(
            torch=torch,
            state=state,
            clip_value=5.0,
        )

        self.assertEqual(report["status"], "clipper_unavailable")
        self.assertFalse(report["applied"])

    def test_value_clip_skips_when_disabled(self) -> None:
        _fixture, state, torch = _state_with_first_gradient(9.0)

        report = apply_torch_gradient_value_clip(
            torch=torch,
            state=state,
            clip_value=0.0,
        )

        self.assertEqual(report["status"], "not_required")
        self.assertFalse(report["applied"])


def _state_with_first_gradient(value: float) -> tuple[dict, dict, object]:
    fixture = _scalar_training_fixture()
    importer = fake_torch_importer(training_runtime=True)
    torch = importer("torch")
    state = build_torch_training_state(
        fixture=fixture,
        torch=torch,
        runtime=torch_runtime_status(importer=importer),
    )
    first = state["parameters"][0]
    first["tensor"].grad = torch.tensor(_filled(first["shape"], value))
    return fixture, state, torch


def _filled(shape: list[int], value: float) -> object:
    if not shape:
        return value
    return [_filled(shape[1:], value) for _index in range(shape[0])]


def _scalar_training_fixture() -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="tiny-training-scalar",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=OptimizationConfig(optimizer="adamw"),
        learning_rate=0.02,
        steps=1,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
