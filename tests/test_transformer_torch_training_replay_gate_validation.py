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
from transformer_torch_training_replay_gate_check_validation import (  # noqa: E402
    BOOLEAN_REPLAY_GATE_CHECKS,
    REPLAY_CONTROL_COUNT_CHECK,
    REPLAY_PROBE_GATE_CHECKS,
    STATUS_REPLAY_GATE_CHECKS,
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

    def test_validator_rejects_stale_bool_check_payload(self) -> None:
        gate = _gate()
        _check(gate, "runtime_available")["actual"] = False

        with self.assertRaisesRegex(ValueError, "runtime_available.passed"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_stale_status_check_payload(self) -> None:
        gate = _gate()
        _check(gate, "runtime_kind")["actual"] = "test_double"

        with self.assertRaisesRegex(ValueError, "runtime_kind.passed"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_stale_replay_count_payload(self) -> None:
        gate = _gate()
        _check(gate, "replay_gradient_signatures")["executed_count"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "replay_gradient_signatures.passed",
        ):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_stale_replay_probe_payload(self) -> None:
        gate = _gate()
        _check(gate, "replay_buffer")["probe_passed"] = False

        with self.assertRaisesRegex(ValueError, "replay_buffer.passed"):
            validate_torch_training_replay_parity_gate(gate)

    def test_validator_rejects_replay_probe_proof_flag_drift(self) -> None:
        gate = _gate()
        _check(gate, "replay_buffer")["proof_flags"] = {}

        with self.assertRaisesRegex(ValueError, "replay_buffer.proof_flags"):
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
    failed_checks = list(failed_checks or [])
    failed = set(failed_checks)
    checks = [
        _gate_check(name, name in failed)
        for name in TORCH_TRAINING_REPLAY_GATE_CHECKS
    ]
    return {
        "schema_version": 1,
        "status": status,
        "passed": not failed_checks,
        "parity_status": parity_status,
        "implementation_status": implementation_status,
        "promoted_training_backend": False,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_check_count": len(checks) - len(failed_checks),
            "failed_checks": failed_checks,
        },
        "reason": "one or more replay parity gates have not matched",
    }


def _gate_check(name: str, failed: bool) -> dict:
    if name in BOOLEAN_REPLAY_GATE_CHECKS:
        return {"name": name, "passed": not failed, "actual": not failed}
    if name in STATUS_REPLAY_GATE_CHECKS:
        expected = STATUS_REPLAY_GATE_CHECKS[name]
        actual = "stale" if failed else expected
        return {
            "name": name,
            "passed": not failed,
            "expected": expected,
            "actual": actual,
        }
    if name == REPLAY_CONTROL_COUNT_CHECK:
        return _replay_control_count_check(failed)
    if name in REPLAY_PROBE_GATE_CHECKS:
        return _replay_probe_check(name, failed)
    raise AssertionError(name)


def _replay_control_count_check(failed: bool) -> dict:
    planned = 2
    executed = 1 if failed else planned
    return {
        "name": REPLAY_CONTROL_COUNT_CHECK,
        "passed": not failed,
        "schema_version": 1,
        "expected_schema_version": 1,
        "count_types_valid": True,
        "match_count": planned,
        "mismatch_count": 0,
        "planned_count": planned,
        "executed_count": executed,
        "backward_count": planned,
        "microstep_count": planned,
    }


def _replay_probe_check(name: str, failed: bool) -> dict:
    expected_status, expected_schema, proof_flags = REPLAY_PROBE_GATE_CHECKS[name]
    return {
        "name": name,
        "passed": not failed,
        "probe_passed": not failed,
        "expected": expected_status,
        "status": expected_status,
        "schema_version": expected_schema,
        "expected_schema_version": expected_schema,
        "proof_flags": {flag: True for flag in proof_flags},
    }


def _check(gate: dict, name: str) -> dict:
    return next(check for check in gate["checks"] if check["name"] == name)


if __name__ == "__main__":
    unittest.main()
