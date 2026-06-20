from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_audit_status import (
    validate_torch_training_parity_attempt_audit_status,
)


class TransformerTorchTrainingParityAttemptAuditStatusTests(unittest.TestCase):
    def test_validator_accepts_runtime_blocked_audit(self) -> None:
        validate_torch_training_parity_attempt_audit_status(_runtime_blocked())

    def test_validator_rejects_stale_runtime_blocked_status(self) -> None:
        audit = _runtime_blocked()
        audit["attempt_status"] = "training_parity_matched"

        with self.assertRaisesRegex(ValueError, "attempt_status"):
            validate_torch_training_parity_attempt_audit_status(audit)

    def test_validator_rejects_stale_runtime_blocked_passed_flag(self) -> None:
        audit = _runtime_blocked()
        audit["attempt_passed"] = True

        with self.assertRaisesRegex(ValueError, "attempt_passed"):
            validate_torch_training_parity_attempt_audit_status(audit)

    def test_validator_accepts_replay_pending_audit(self) -> None:
        validate_torch_training_parity_attempt_audit_status(
            {
                "attempt_status": "training_replay_parity_pending",
                "attempt_passed": False,
                "runtime_status": "passed",
                "parity_attempt_allowed": True,
                "training_replay_parity_status": "training_replay_parity_pending",
                "training_replay_parity_passed": False,
                "training_report_passed": True,
            }
        )

    def test_validator_accepts_matched_audit(self) -> None:
        validate_torch_training_parity_attempt_audit_status(
            {
                "attempt_status": "training_parity_matched",
                "attempt_passed": True,
                "runtime_status": "passed",
                "parity_attempt_allowed": True,
                "training_replay_parity_status": "training_replay_parity_matched",
                "training_replay_parity_passed": True,
                "training_report_passed": True,
            }
        )

    def test_validator_rejects_report_bypass(self) -> None:
        audit = _runtime_blocked()
        audit["training_report_passed"] = True
        audit["attempt_passed"] = True

        with self.assertRaisesRegex(ValueError, "attempt_passed"):
            validate_torch_training_parity_attempt_audit_status(audit)


def _runtime_blocked() -> dict:
    return {
        "attempt_status": "blocked_runtime_unavailable",
        "attempt_passed": False,
        "runtime_status": "blocked_runtime_unavailable",
        "parity_attempt_allowed": False,
        "training_replay_parity_status": "training_replay_parity_pending",
        "training_replay_parity_passed": False,
        "training_report_passed": False,
    }


if __name__ == "__main__":
    unittest.main()
