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


class TransformerTorchTrainingParityAttemptPassGuardTests(unittest.TestCase):
    def test_validator_rejects_stale_attempt_passed_flag(self) -> None:
        attempt = _attempt()
        attempt["passed"] = True

        with self.assertRaisesRegex(ValueError, "passed flag"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_report_passed_runtime_bypass(self) -> None:
        attempt = _attempt()
        attempt["training_parity_report"]["passed"] = True
        attempt["passed"] = True

        with self.assertRaisesRegex(ValueError, "passed flag"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_report_passed_replay_bypass(self) -> None:
        attempt = _attempt()
        attempt["runtime"]["parity_attempt_allowed"] = True
        attempt["runtime"]["passed"] = True
        attempt["runtime"]["status"] = "ready_for_pytorch_parity"
        attempt["candidate"]["training_readiness_status"] = "ready"
        attempt["training_replay_parity_gate"]["status"] = (
            "training_replay_parity_pending"
        )
        attempt["training_replay_parity_gate"]["passed"] = False
        attempt["training_parity_report"]["passed"] = True
        attempt["passed"] = True
        attempt["status"] = "training_replay_parity_pending"
        attempt["next_requirements"]["stage"] = "training_replay_parity"
        attempt["next_requirements"]["status"] = "pending"
        attempt["next_requirements"]["runtime_status"] = "passed"
        attempt["next_requirements"]["parity_attempt_allowed"] = True
        attempt["next_requirements"]["training_readiness_status"] = "ready"
        attempt["next_requirements"]["training_replay_parity_status"] = (
            "training_replay_parity_pending"
        )
        attempt["next_requirements"]["training_report_passed"] = True

        with self.assertRaisesRegex(ValueError, "passed flag"):
            validate_torch_training_parity_attempt(attempt)


def _attempt() -> dict:
    artifacts = build_torch_training_parity_attempt(
        corpus_dir=ROOT / "corpus",
        fixture_id="pass-guard-training-parity-attempt",
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


if __name__ == "__main__":
    unittest.main()
