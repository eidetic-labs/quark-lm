from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from support.fake_torch import fake_torch_importer
from transformer_model import OptimizationConfig
from transformer_torch_backend import (
    TORCH_TRAINING_BLOCKED_STATUS,
    TORCH_TRAINING_PENDING_STATUS,
    TORCH_TRAINING_READY_STATUS,
    build_torch_training_readiness,
    torch_runtime_status,
    validate_torch_training_readiness,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchTrainingReadinessTests(unittest.TestCase):
    def test_readiness_blocks_missing_runtime(self) -> None:
        fixture = _scalar_training_fixture()
        runtime = torch_runtime_status(importer=_missing_importer)

        readiness = build_torch_training_readiness(
            fixture=fixture,
            runtime=runtime,
            importer=_missing_importer,
        )

        self.assertEqual(readiness["status"], TORCH_TRAINING_BLOCKED_STATUS)
        self.assertIn("runtime_available", readiness["summary"]["failed_checks"])
        validate_torch_training_readiness(readiness)

    def test_readiness_marks_forward_only_runtime_pending(self) -> None:
        fixture = _scalar_training_fixture()
        runtime = torch_runtime_status(importer=fake_torch_importer())

        readiness = build_torch_training_readiness(
            fixture=fixture,
            runtime=runtime,
            importer=fake_torch_importer(),
        )

        self.assertEqual(readiness["status"], TORCH_TRAINING_PENDING_STATUS)
        self.assertIn("autograd", readiness["summary"]["failed_checks"])
        self.assertIn("adamw_optimizer", readiness["summary"]["failed_checks"])
        self.assertNotIn("parameter_manifest", readiness["summary"]["failed_checks"])
        validate_torch_training_readiness(readiness)

    def test_readiness_accepts_training_capable_runtime(self) -> None:
        fixture = _scalar_training_fixture()
        importer = fake_torch_importer(training_runtime=True)
        runtime = torch_runtime_status(importer=importer)

        readiness = build_torch_training_readiness(
            fixture=fixture,
            runtime=runtime,
            importer=importer,
        )

        self.assertEqual(readiness["status"], TORCH_TRAINING_READY_STATUS)
        self.assertEqual(readiness["summary"]["failed_checks"], [])
        validate_torch_training_readiness(readiness)


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


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
