from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import transformer_torch_training_parity_attempt_writer as writer_module
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
from transformer_torch_training_parity_attempt_reader import (
    load_torch_training_parity_attempt_artifact_set,
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

    def test_loader_validates_written_artifact_set(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            write_torch_training_parity_attempt(Path(temp), artifacts)

            loaded = load_torch_training_parity_attempt_artifact_set(Path(temp))

        self.assertEqual(
            loaded["attempt"]["fixture_id"],
            artifacts["attempt"]["fixture_id"],
        )

    def test_loader_rejects_disk_payload_drift(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            written = write_torch_training_parity_attempt(Path(temp), artifacts)
            candidate_path = Path(written["artifacts"]["candidate"])
            candidate = _read_json(candidate_path)
            candidate["unvalidated_extra_field"] = "drift"
            _write_json(candidate_path, candidate)

            with self.assertRaisesRegex(ValueError, "artifact_hashes"):
                load_torch_training_parity_attempt_artifact_set(Path(temp))

    def test_loader_rejects_stale_recorded_artifact_path(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            written = write_torch_training_parity_attempt(Path(temp), artifacts)
            attempt_path = Path(written["artifacts"]["attempt"])
            attempt = _read_json(attempt_path)
            attempt["artifacts"]["candidate"] = "elsewhere.json"
            _write_json(attempt_path, attempt)

            with self.assertRaisesRegex(ValueError, "artifacts.candidate"):
                load_torch_training_parity_attempt_artifact_set(Path(temp))

    def test_writer_rejects_mixed_artifact_set(self) -> None:
        artifacts = _artifacts()
        other_artifacts = _artifacts(fixture_id="other-training-parity-attempt")
        artifacts["candidate"] = other_artifacts["candidate"]

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "candidate.fixture_id"):
                write_torch_training_parity_attempt(Path(temp), artifacts)

    def test_writer_reloads_and_rejects_corrupt_written_payload(self) -> None:
        artifacts = _artifacts()
        original_write = writer_module.write_json_artifact

        def corrupting_write(path: Path, payload: dict) -> None:
            if path.name == "torch_training_candidate.json":
                payload = {**payload, "unvalidated_extra_field": "drift"}
            original_write(path, payload)

        writer_module.write_json_artifact = corrupting_write
        try:
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, "artifact_hashes"):
                    writer_module.write_torch_training_parity_attempt(
                        Path(temp),
                        artifacts,
                    )
        finally:
            writer_module.write_json_artifact = original_write


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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
