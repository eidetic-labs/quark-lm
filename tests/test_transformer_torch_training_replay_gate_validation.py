from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
    TORCH_TRAINING_REPLAY_GATE_CHECKS,
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
    validate_torch_training_replay_parity_gate,
)


class TransformerTorchTrainingReplayGateValidationTests(unittest.TestCase):
    def test_validator_accepts_matched_gate(self) -> None:
        validate_torch_training_replay_parity_gate(_gate())

    def test_validator_accepts_pending_gate(self) -> None:
        validate_torch_training_replay_parity_gate(
            _gate(
                status=TORCH_TRAINING_REPLAY_PENDING_STATUS,
                parity_status="pending",
                implementation_status=TORCH_TRAINING_REPLAY_PENDING_STATUS,
                failed_checks=["replay_buffer"],
            )
        )

    def test_validator_accepts_blocked_gate(self) -> None:
        validate_torch_training_replay_parity_gate(
            _gate(
                status=TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
                parity_status="failed",
                implementation_status="runtime_unavailable",
                failed_checks=["runtime_available"],
            )
        )

    def test_validator_rejects_stale_passed_flag(self) -> None:
        gate = _gate(failed_checks=["replay_buffer"])
        gate["passed"] = True

        with self.assertRaisesRegex(ValueError, "passed"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_stale_status(self) -> None:
        gate = _gate(failed_checks=["replay_buffer"])
        gate["status"] = TORCH_TRAINING_REPLAY_MATCHED_STATUS

        with self.assertRaisesRegex(ValueError, "status"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_stale_parity_status(self) -> None:
        gate = _gate(
            status=TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
            parity_status="failed",
            implementation_status="runtime_unavailable",
            failed_checks=["runtime_available"],
        )
        gate["parity_status"] = "pending"

        with self.assertRaisesRegex(ValueError, "parity_status"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_boolean_summary_counts(self) -> None:
        gate = _gate()
        gate["summary"]["check_count"] = True

        with self.assertRaisesRegex(ValueError, "check_count"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_stale_failed_checks(self) -> None:
        gate = _gate()
        gate["summary"]["failed_checks"] = ["replay_buffer"]

        with self.assertRaisesRegex(ValueError, "failed_checks"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_unknown_check_catalog(self) -> None:
        gate = _gate()
        gate["checks"].append({"name": "outside_gate", "passed": True})
        gate["summary"]["check_count"] += 1
        gate["summary"]["passed_check_count"] += 1

        with self.assertRaisesRegex(ValueError, "catalog"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_non_bool_check_passed(self) -> None:
        gate = _gate()
        gate["checks"][0]["passed"] = 1

        with self.assertRaisesRegex(ValueError, "runtime_available.passed"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_promotion(self) -> None:
        gate = _gate()
        gate["promoted_training_backend"] = True

        with self.assertRaisesRegex(ValueError, "must not promote"):
            validate_torch_training_replay_parity_gate(gate)

    def test_check_catalog_is_public(self) -> None:
        self.assertEqual(TORCH_TRAINING_REPLAY_GATE_CHECKS[0], "runtime_available")
        self.assertEqual(TORCH_TRAINING_REPLAY_GATE_CHECKS[-1], "replay_checkpoint")


def _gate(
    *,
    status: str = TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    parity_status: str = "matched",
    implementation_status: str = TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    failed_checks: list[str] | None = None,
) -> dict:
    failed = set(failed_checks or [])
    checks = [
        {"name": name, "passed": name not in failed}
        for name in TORCH_TRAINING_REPLAY_GATE_CHECKS
    ]
    return {
        "schema_version": 1,
        "status": status,
        "passed": not failed,
        "parity_status": parity_status,
        "implementation_status": implementation_status,
        "promoted_training_backend": False,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_check_count": len(checks) - len(failed),
            "failed_checks": list(failed_checks or []),
        },
        "reason": "one or more replay parity gates have not matched",
    }


if __name__ == "__main__":
    unittest.main()
