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
from transformer_torch_training_parity_attempt_artifact_set import (
    validate_torch_training_parity_attempt_artifact_set,
)
from transformer_torch_training_parity_attempt_hashes import (
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
    build_torch_training_parity_attempt_hashes,
)
from transformer_torch_training_parity_attempt_summaries import (
    build_torch_attempt_candidate_summary,
)


class TransformerTorchTrainingParityAttemptArtifactSetTests(unittest.TestCase):
    def test_valid_artifact_set_passes(self) -> None:
        validate_torch_training_parity_attempt_artifact_set(_artifacts())

    def test_validator_rejects_mixed_candidate_fixture_id(self) -> None:
        artifacts = _artifacts()
        artifacts["candidate"]["fixture_id"] = "different-fixture"

        with self.assertRaisesRegex(ValueError, "candidate.fixture_id"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_validator_rejects_stale_attempt_corpus_hash(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["corpus"]["train_sha256"] = "stale"

        with self.assertRaisesRegex(ValueError, "corpus.train_sha256"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_validator_rejects_stale_candidate_corpus_hash(self) -> None:
        artifacts = _artifacts()
        artifacts["candidate"]["backend"]["corpus_hash"] = "stale"

        with self.assertRaisesRegex(ValueError, "candidate.backend.corpus_hash"):
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
        artifacts["attempt"]["candidate"] = build_torch_attempt_candidate_summary(
            artifacts["candidate"]
        )

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
