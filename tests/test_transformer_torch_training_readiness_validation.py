from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_readiness import (
    TORCH_TRAINING_BLOCKED_STATUS,
    TORCH_TRAINING_PENDING_STATUS,
    TORCH_TRAINING_READY_STATUS,
)
from transformer_torch_training_readiness_validation import (
    TORCH_TRAINING_READINESS_BASE_CHECKS,
    TORCH_TRAINING_READINESS_CHECK_CATALOGS,
    TORCH_TRAINING_READINESS_RUNTIME_CHECKS,
    validate_torch_training_readiness,
)


class TransformerTorchTrainingReadinessValidationTests(unittest.TestCase):
    def test_validator_accepts_ready_runtime_catalog(self) -> None:
        validate_torch_training_readiness(_ready_readiness())

    def test_validator_accepts_blocked_base_catalog(self) -> None:
        validate_torch_training_readiness(_blocked_readiness())

    def test_validator_rejects_stale_status(self) -> None:
        readiness = _blocked_readiness()
        readiness["status"] = TORCH_TRAINING_READY_STATUS

        with self.assertRaisesRegex(ValueError, "status"):
            validate_torch_training_readiness(readiness)

    def test_validator_rejects_stale_failed_checks(self) -> None:
        readiness = _ready_readiness()
        readiness["summary"]["failed_checks"] = ["autograd"]

        with self.assertRaisesRegex(ValueError, "failed_checks"):
            validate_torch_training_readiness(readiness)

    def test_validator_rejects_boolean_summary_counts(self) -> None:
        readiness = _ready_readiness()
        readiness["summary"]["check_count"] = True

        with self.assertRaisesRegex(ValueError, "check_count"):
            validate_torch_training_readiness(readiness)

    def test_validator_rejects_non_bool_check_passed(self) -> None:
        readiness = _ready_readiness()
        readiness["checks"][0]["passed"] = 1

        with self.assertRaisesRegex(ValueError, "runtime_available.passed"):
            validate_torch_training_readiness(readiness)

    def test_validator_rejects_unknown_check_catalog(self) -> None:
        readiness = _ready_readiness()
        readiness["checks"].append({"name": "outside_capability", "passed": True})
        readiness["summary"]["check_count"] += 1
        readiness["summary"]["passed_check_count"] += 1

        with self.assertRaisesRegex(ValueError, "catalog"):
            validate_torch_training_readiness(readiness)

    def test_validator_accepts_pending_runtime_catalog(self) -> None:
        readiness = _ready_readiness()
        readiness["status"] = TORCH_TRAINING_PENDING_STATUS
        readiness["checks"][-1]["passed"] = False
        readiness["summary"]["passed_check_count"] = len(readiness["checks"]) - 1
        readiness["summary"]["failed_checks"] = ["adamw_optimizer"]

        validate_torch_training_readiness(readiness)

    def test_check_catalogs_are_public_contracts(self) -> None:
        self.assertEqual(
            TORCH_TRAINING_READINESS_BASE_CHECKS,
            ("runtime_available", "dtype_available", "parameter_manifest"),
        )
        self.assertIn(
            TORCH_TRAINING_READINESS_BASE_CHECKS + TORCH_TRAINING_READINESS_RUNTIME_CHECKS,
            TORCH_TRAINING_READINESS_CHECK_CATALOGS,
        )


def _blocked_readiness() -> dict:
    checks = [
        {"name": "runtime_available", "passed": False},
        {"name": "dtype_available", "passed": False},
        {"name": "parameter_manifest", "passed": True},
    ]
    return _readiness(TORCH_TRAINING_BLOCKED_STATUS, checks)


def _ready_readiness() -> dict:
    checks = [
        {"name": "runtime_available", "passed": True},
        {"name": "dtype_available", "passed": True},
        {"name": "parameter_manifest", "passed": True},
        {"name": "torch_tensor", "passed": True},
        {"name": "autograd", "passed": True},
        {"name": "adamw_optimizer", "passed": True},
    ]
    return _readiness(TORCH_TRAINING_READY_STATUS, checks)


def _readiness(status: str, checks: list[dict]) -> dict:
    failed = [check["name"] for check in checks if check["passed"] is not True]
    return {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_check_count": len(checks) - len(failed),
            "failed_checks": failed,
        },
    }


if __name__ == "__main__":
    unittest.main()
