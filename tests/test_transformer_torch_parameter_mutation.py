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
    TORCH_PARAMETER_MUTATION_NOT_OBSERVED_STATUS,
    TORCH_PARAMETER_MUTATION_OBSERVED_STATUS,
    build_torch_parameter_mutation_report,
    build_torch_training_state,
    snapshot_torch_parameters,
    torch_runtime_status,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchParameterMutationTests(unittest.TestCase):
    def test_report_detects_parameter_mutation(self) -> None:
        state = _training_state()
        before = snapshot_torch_parameters(state)
        state["parameters"][0]["tensor"].value[0][0] += 0.25

        report = build_torch_parameter_mutation_report(
            before=before,
            after=snapshot_torch_parameters(state),
        )

        self.assertEqual(report["status"], TORCH_PARAMETER_MUTATION_OBSERVED_STATUS)
        self.assertEqual(report["changed_scalar_count"], 1)
        self.assertEqual(report["changed_tensor_count"], 1)
        self.assertAlmostEqual(report["max_abs_delta"], 0.25)
        json.dumps(report)

    def test_report_marks_unchanged_parameters(self) -> None:
        state = _training_state()
        before = snapshot_torch_parameters(state)

        report = build_torch_parameter_mutation_report(
            before=before,
            after=snapshot_torch_parameters(state),
        )

        self.assertEqual(
            report["status"],
            TORCH_PARAMETER_MUTATION_NOT_OBSERVED_STATUS,
        )
        self.assertEqual(report["changed_scalar_count"], 0)


def _training_state() -> dict:
    fixture = _scalar_training_fixture()
    importer = fake_torch_importer(training_runtime=True)
    return build_torch_training_state(
        fixture=fixture,
        torch=importer("torch"),
        runtime=torch_runtime_status(importer=importer),
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
        optimizer_config=OptimizationConfig(optimizer="adamw"),
        learning_rate=0.02,
        steps=1,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
