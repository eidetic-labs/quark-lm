from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_validation import (
    validate_torch_training_parity_attempt,
)


class TransformerTorchTrainingParityAttemptValidationTests(unittest.TestCase):
    def test_valid_attempt_summary_passes(self) -> None:
        validate_torch_training_parity_attempt(_attempt())

    def test_validator_rejects_missing_promotion_gate(self) -> None:
        attempt = _attempt()
        attempt.pop("training_backend_promotion_gate")

        with self.assertRaisesRegex(ValueError, "training_backend_promotion_gate"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_accidental_backend_promotion(self) -> None:
        attempt = _attempt()
        attempt["promoted_training_backend"] = True

        with self.assertRaisesRegex(ValueError, "must not promote"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_dirty_closed_world_boundary(self) -> None:
        attempt = _attempt()
        attempt["closed_world_boundary"]["pretrained_tokenizer_imported"] = True

        with self.assertRaisesRegex(ValueError, "pretrained_tokenizer_imported"):
            validate_torch_training_parity_attempt(attempt)

    def test_writer_validation_requires_artifact_paths(self) -> None:
        attempt = _attempt()

        with self.assertRaisesRegex(ValueError, "artifacts"):
            validate_torch_training_parity_attempt(
                attempt,
                require_artifacts=True,
            )

        attempt["artifacts"] = {
            "fixture": "fixture.json",
            "candidate": "candidate.json",
            "report": "report.json",
            "attempt": "attempt.json",
        }
        validate_torch_training_parity_attempt(
            attempt,
            require_artifacts=True,
        )


def _attempt() -> dict:
    artifacts = build_torch_training_parity_attempt(
        corpus_dir=ROOT / "corpus",
        fixture_id="validation-training-parity-attempt",
        seed=53,
        context_index=4,
        context_size=4,
        embedding_dim=4,
        feedforward_dim=8,
        steps=2,
        importer=_missing_importer,
    )
    return copy.deepcopy(artifacts["attempt"])


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)
