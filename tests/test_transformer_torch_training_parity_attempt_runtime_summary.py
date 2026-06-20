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
    build_torch_runtime_report_hash,
    build_torch_training_attempt_payload_hash,
)
from transformer_torch_training_parity_attempt_summary_validation import (
    validate_torch_training_parity_attempt_summaries,
)


class TransformerTorchTrainingParityAttemptRuntimeSummaryTests(unittest.TestCase):
    def test_attempt_runtime_summary_records_runtime_report_hash(self) -> None:
        artifacts = _artifacts()
        attempt = artifacts["attempt"]
        candidate = artifacts["candidate"]
        report = artifacts["report"]

        self.assertEqual(
            attempt["runtime"]["runtime_report_sha256"],
            build_torch_runtime_report_hash(candidate["runtime_report"]),
        )
        self.assertEqual(
            attempt["runtime"]["failed_checks"],
            candidate["runtime_report"]["summary"]["failed_checks"],
        )
        self.assertEqual(
            attempt["candidate"]["candidate_sha256"],
            build_torch_training_attempt_payload_hash(candidate),
        )
        self.assertEqual(
            attempt["candidate"]["training_readiness_failed_checks"],
            candidate["training_readiness"]["summary"]["failed_checks"],
        )
        self.assertEqual(
            attempt["training_replay_parity_gate"][
                "training_replay_parity_gate_sha256"
            ],
            build_torch_training_attempt_payload_hash(
                candidate["training_replay_parity_gate"]
            ),
        )
        self.assertEqual(
            attempt["training_parity_report"]["training_parity_report_sha256"],
            build_torch_training_attempt_payload_hash(report),
        )
        validate_torch_training_parity_attempt_summaries(artifacts["attempt"])
        validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_summary_validator_rejects_malformed_runtime_report_hash(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["runtime"]["runtime_report_sha256"] = "not-a-hash"

        with self.assertRaisesRegex(ValueError, "runtime_report_sha256"):
            validate_torch_training_parity_attempt_summaries(artifacts["attempt"])

    def test_artifact_set_rejects_stale_runtime_report_hash(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["runtime"]["runtime_report_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "attempt.runtime"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_artifact_set_rejects_stale_candidate_hash(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["candidate"]["candidate_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "attempt.candidate"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_artifact_set_rejects_stale_replay_gate_hash(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["training_replay_parity_gate"][
            "training_replay_parity_gate_sha256"
        ] = "0" * 64

        with self.assertRaisesRegex(ValueError, "training_replay_parity_gate"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)

    def test_artifact_set_rejects_stale_report_hash(self) -> None:
        artifacts = _artifacts()
        artifacts["attempt"]["training_parity_report"][
            "training_parity_report_sha256"
        ] = "0" * 64

        with self.assertRaisesRegex(ValueError, "training_parity_report"):
            validate_torch_training_parity_attempt_artifact_set(artifacts)


def _artifacts() -> dict:
    return copy.deepcopy(
        build_torch_training_parity_attempt(
            corpus_dir=ROOT / "corpus",
            fixture_id="runtime-summary-training-parity-attempt",
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


if __name__ == "__main__":
    unittest.main()
