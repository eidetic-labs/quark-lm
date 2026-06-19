from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.fake_torch import fake_torch_importer
from transformer_torch_training_parity_attempt import (
    TORCH_TRAINING_PARITY_ATTEMPT_KIND,
    build_torch_training_parity_attempt,
    write_torch_training_parity_attempt,
)


class TransformerTorchTrainingParityAttemptTests(unittest.TestCase):
    def test_attempt_records_missing_runtime_without_training_claim(self) -> None:
        artifacts = _attempt(importer=_missing_importer)
        attempt = artifacts["attempt"]

        self.assertEqual(attempt["kind"], TORCH_TRAINING_PARITY_ATTEMPT_KIND)
        self.assertEqual(attempt["status"], "blocked_runtime_unavailable")
        self.assertFalse(attempt["passed"])
        self.assertFalse(attempt["promoted_training_backend"])
        self.assertEqual(
            attempt["closed_world_boundary"]["training_text_source"],
            "admitted_curriculum",
        )
        self.assertFalse(
            attempt["closed_world_boundary"]["pretrained_weights_imported"]
        )
        self.assertEqual(
            attempt["candidate"]["implementation_status"],
            "runtime_unavailable",
        )
        self.assertIn(
            "runtime_available",
            artifacts["candidate"]["training_readiness"]["summary"][
                "failed_checks"
            ],
        )

    def test_attempt_records_test_double_as_blocked_but_keeps_gate_detail(self) -> None:
        artifacts = _attempt(
            importer=fake_torch_importer(
                training_runtime=True,
                gradient_runtime=True,
            )
        )
        attempt = artifacts["attempt"]

        self.assertEqual(attempt["status"], "blocked_test_double_runtime")
        self.assertFalse(attempt["runtime"]["parity_attempt_allowed"])
        self.assertEqual(attempt["runtime"]["runtime_kind"], "test_double")
        self.assertIn(
            "runtime_kind",
            attempt["training_replay_parity_gate"]["failed_checks"],
        )

    def test_writer_outputs_attempt_artifact_set(self) -> None:
        artifacts = _attempt(importer=_missing_importer)
        with tempfile.TemporaryDirectory() as temp:
            written = write_torch_training_parity_attempt(Path(temp), artifacts)
            artifact_paths = {
                name: Path(path)
                for name, path in written["artifacts"].items()
            }

            for path in artifact_paths.values():
                self.assertTrue(path.exists())

            attempt = json.loads(
                artifact_paths["attempt"].read_text(encoding="utf-8")
            )
            fixture = json.loads(
                artifact_paths["fixture"].read_text(encoding="utf-8")
            )

        self.assertEqual(attempt["kind"], TORCH_TRAINING_PARITY_ATTEMPT_KIND)
        self.assertEqual(fixture["fixture_id"], attempt["fixture_id"])


def _attempt(*, importer) -> dict:
    return build_torch_training_parity_attempt(
        corpus_dir=ROOT / "corpus",
        fixture_id="test-training-parity-attempt",
        seed=53,
        context_index=4,
        context_size=4,
        embedding_dim=4,
        feedforward_dim=8,
        steps=2,
        importer=importer,
    )


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)
