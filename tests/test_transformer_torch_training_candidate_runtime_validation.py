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
    TORCH_TRAINING_CANDIDATE_RUNTIME_FIELDS,
    build_torch_training_parity_candidate,
    validate_torch_training_candidate_runtime_report,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchTrainingCandidateRuntimeValidationTests(unittest.TestCase):
    def test_validator_accepts_embedded_runtime_report(self) -> None:
        validate_torch_training_candidate_runtime_report(_candidate())

    def test_validator_rejects_missing_runtime_report(self) -> None:
        candidate = _candidate()
        candidate.pop("runtime_report")

        with self.assertRaisesRegex(ValueError, "candidate.runtime_report"):
            validate_torch_training_candidate_runtime_report(candidate)

    def test_validator_rejects_runtime_payload_drift(self) -> None:
        candidate = _candidate()
        candidate["runtime_report"]["runtime"] = {
            **candidate["runtime_report"]["runtime"],
            "dtype": "float16",
        }

        with self.assertRaisesRegex(ValueError, "runtime_report.runtime"):
            validate_torch_training_candidate_runtime_report(candidate)

    def test_validator_rejects_dirty_runtime_boundary(self) -> None:
        candidate = _candidate()
        candidate["runtime_report"]["closed_world_boundary"][
            "pretrained_weights_imported"
        ] = True

        with self.assertRaisesRegex(ValueError, "pretrained_weights_imported"):
            validate_torch_training_candidate_runtime_report(candidate)

    def test_runtime_field_catalog_is_public(self) -> None:
        self.assertEqual(
            TORCH_TRAINING_CANDIDATE_RUNTIME_FIELDS,
            ("runtime", "runtime_report"),
        )


def _candidate() -> dict:
    return build_torch_training_parity_candidate(
        fixture=_scalar_training_fixture(),
        importer=fake_torch_importer(
            training_runtime=True,
            gradient_runtime=True,
        ),
    )


def _scalar_training_fixture() -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="candidate-runtime-validation-scalar",
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
