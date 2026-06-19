from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
    write_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_artifact_set import (
    validate_torch_training_parity_attempt_artifact_set,
)


class TransformerTorchTrainingParityAttemptArtifactSetTests(unittest.TestCase):
    def test_valid_artifact_set_passes(self) -> None:
        validate_torch_training_parity_attempt_artifact_set(_artifacts())

    def test_validator_rejects_mixed_candidate_fixture_id(self) -> None:
        artifacts = _artifacts()
        artifacts["candidate"]["fixture_id"] = "different-fixture"

        with self.assertRaisesRegex(ValueError, "candidate.fixture_id"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_validator_rejects_stale_candidate_summary(self) -> None:
        artifacts = _artifacts()
        artifacts["candidate"]["implementation_status"] = "training_replay_pending"

        with self.assertRaisesRegex(ValueError, "attempt.candidate"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_validator_rejects_stale_report_summary(self) -> None:
        artifacts = _artifacts()
        artifacts["report"]["passed"] = True

        with self.assertRaisesRegex(ValueError, "training_parity_report"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_writer_rejects_mixed_artifact_set(self) -> None:
        artifacts = _artifacts()
        other_artifacts = _artifacts(fixture_id="other-training-parity-attempt")
        artifacts["candidate"] = other_artifacts["candidate"]

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "candidate.fixture_id"):
                write_torch_training_parity_attempt(Path(temp), artifacts)


def _artifacts(
    *,
    fixture_id: str = "artifact-set-training-parity-attempt",
) -> dict:
    return copy.deepcopy(
        build_torch_training_parity_attempt(
            corpus_dir=ROOT / "corpus",
            fixture_id=fixture_id,
            seed=53,
            context_index=4,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            steps=2,
            importer=_missing_importer,
        )
    )


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)
