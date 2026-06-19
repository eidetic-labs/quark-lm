from __future__ import annotations

import copy
import json
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
from transformer_torch_training_parity_attempt_hashes import (
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
    build_torch_training_parity_attempt_hashes,
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

    def test_validator_rejects_report_not_rebuilt_from_candidate(self) -> None:
        artifacts = _artifacts()
        artifacts["candidate"]["training_case"]["final_loss"] = 0.125

        with self.assertRaisesRegex(ValueError, "artifacts.report"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_validator_rejects_promotion_gate_not_rebuilt_from_payloads(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["training_backend_promotion_gate"][
            "parity_evidence_matched"
        ] = True

        with self.assertRaisesRegex(ValueError, "training_backend_promotion_gate"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_validator_rejects_next_requirements_not_rebuilt_from_payloads(
        self,
    ) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["next_requirements"]["next_actions"] = []

        with self.assertRaisesRegex(ValueError, "next_requirements"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_validator_accepts_matching_artifact_hashes(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"][
            "artifact_hash_algorithm"
        ] = TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM
        artifacts["attempt"]["artifact_hashes"] = (
            build_torch_training_parity_attempt_hashes(artifacts)
        )

        validate_torch_training_parity_attempt_artifact_set(
            artifacts,
            require_artifact_hashes=True,
        )

    def test_validator_requires_artifact_hashes_when_requested(self) -> None:
        artifacts = _artifacts()

        with self.assertRaisesRegex(ValueError, "artifact_hash_algorithm"):
            validate_torch_training_parity_attempt_artifact_set(
                artifacts,
                require_artifact_hashes=True,
            )

    def test_validator_rejects_stale_artifact_hashes(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"][
            "artifact_hash_algorithm"
        ] = TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM
        artifacts["attempt"]["artifact_hashes"] = (
            build_torch_training_parity_attempt_hashes(artifacts)
        )
        artifacts["attempt"]["artifact_hashes"]["candidate"] = "stale"

        with self.assertRaisesRegex(ValueError, "artifact_hashes"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_writer_records_payload_hashes(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            written = write_torch_training_parity_attempt(Path(temp), artifacts)
            paths = {name: Path(path) for name, path in written["artifacts"].items()}
            persisted = {
                name: _read_json(paths[name])
                for name in ("fixture", "candidate", "report", "attempt")
            }

        expected_hashes = build_torch_training_parity_attempt_hashes(persisted)
        self.assertEqual(
            persisted["attempt"]["artifact_hash_algorithm"],
            TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
        )
        self.assertEqual(persisted["attempt"]["artifact_hashes"], expected_hashes)
        validate_torch_training_parity_attempt_artifact_set(
            persisted,
            require_artifact_paths=True,
            require_artifact_hashes=True,
        )

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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
