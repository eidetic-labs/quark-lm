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
    build_torch_training_initial_loss_probe,
    build_torch_training_state,
    summarize_torch_training_state,
    torch_training_weights_from_state,
    torch_runtime_status,
    validate_torch_training_state_summary,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchTrainingStateTests(unittest.TestCase):
    def test_state_builds_trainable_tensors_from_parameter_manifest(self) -> None:
        fixture = _scalar_training_fixture()
        importer = fake_torch_importer(training_runtime=True)
        state = build_torch_training_state(
            fixture=fixture,
            torch=importer("torch"),
            runtime=torch_runtime_status(importer=importer),
        )

        summary = summarize_torch_training_state(state)

        validate_torch_training_state_summary(
            summary,
            fixture["parameter_manifest"],
        )
        self.assertEqual(
            summary["parameter_count"],
            fixture["parameter_manifest"]["parameter_count"],
        )
        self.assertEqual(summary["parameters"][0]["name"], "token_embeddings")
        self.assertTrue(summary["parameters"][0]["requires_grad"])
        json.dumps(summary)

    def test_state_overlays_trainable_tensors_onto_weight_tree(self) -> None:
        fixture = _scalar_training_fixture()
        importer = fake_torch_importer(training_runtime=True)
        state = build_torch_training_state(
            fixture=fixture,
            torch=importer("torch"),
            runtime=torch_runtime_status(importer=importer),
        )

        weights = torch_training_weights_from_state(fixture=fixture, state=state)

        self.assertTrue(weights["wq"].requires_grad)
        self.assertTrue(weights["token_embeddings"].requires_grad)
        self.assertEqual(weights["wq"].tolist(), fixture["initial_weights"]["wq"])

    def test_initial_loss_probe_matches_scalar_training_fixture(self) -> None:
        fixture = _scalar_training_fixture()
        importer = fake_torch_importer(training_runtime=True)
        runtime = torch_runtime_status(importer=importer)
        state = build_torch_training_state(
            fixture=fixture,
            torch=importer("torch"),
            runtime=runtime,
        )

        probe = build_torch_training_initial_loss_probe(
            fixture=fixture,
            state=state,
            torch=importer("torch"),
            runtime=runtime,
        )

        self.assertEqual(probe["status"], "matched")
        self.assertAlmostEqual(
            probe["initial_loss"],
            fixture["training_case"]["initial_loss"],
        )
        self.assertLessEqual(probe["max_logit_abs_diff"], 1e-9)

    def test_state_resolves_optional_extra_layer_parameter_paths(self) -> None:
        fixture = _scalar_training_fixture(num_layers=2, use_gated_mlp=True)
        importer = fake_torch_importer(training_runtime=True)

        state = build_torch_training_state(
            fixture=fixture,
            torch=importer("torch"),
            runtime=torch_runtime_status(importer=importer),
        )
        names = {parameter["name"] for parameter in state["parameters"]}

        self.assertIn("w_gate", names)
        self.assertIn("extra_layers.0.wq", names)
        self.assertIn("extra_layers.0.w_gate", names)

    def test_state_rejects_weights_that_do_not_match_manifest_shape(self) -> None:
        fixture = copy.deepcopy(_scalar_training_fixture())
        fixture["initial_weights"]["wq"] = [[1.0]]
        importer = fake_torch_importer(training_runtime=True)

        with self.assertRaisesRegex(ValueError, "wq"):
            build_torch_training_state(
                fixture=fixture,
                torch=importer("torch"),
                runtime=torch_runtime_status(importer=importer),
            )


def _scalar_training_fixture(**config_overrides: object) -> dict:
    tokenizer, ids, config, model = char_model_fixture(
        "abc abc\n",
        seed=53,
        **config_overrides,
    )
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
