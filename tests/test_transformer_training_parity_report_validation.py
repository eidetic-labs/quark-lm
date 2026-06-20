from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_training_parity import validate_training_parity_report  # noqa: E402


class TransformerTrainingParityReportValidationTests(unittest.TestCase):
    def test_validator_accepts_matching_report_summary(self) -> None:
        validate_training_parity_report(_report())

    def test_validator_accepts_missing_candidate_backend(self) -> None:
        report = _report()
        report["candidate_backend"] = None

        validate_training_parity_report(report)

    def test_validator_rejects_stale_passed_flag(self) -> None:
        report = _report(failed_checks=["training_final_loss"])
        report["passed"] = True

        with self.assertRaisesRegex(ValueError, "passed"):
            validate_training_parity_report(report)

    def test_validator_rejects_boolean_summary_count(self) -> None:
        report = _report()
        report["summary"]["check_count"] = True

        with self.assertRaisesRegex(ValueError, "check_count"):
            validate_training_parity_report(report)

    def test_validator_rejects_stale_failed_checks(self) -> None:
        report = _report()
        report["summary"]["failed_checks"] = ["training_final_loss"]

        with self.assertRaisesRegex(ValueError, "failed_checks"):
            validate_training_parity_report(report)

    def test_validator_rejects_duplicate_check_names(self) -> None:
        report = _report()
        report["checks"].append(dict(report["checks"][0]))
        report["summary"]["check_count"] += 1
        report["summary"]["passed_check_count"] += 1

        with self.assertRaisesRegex(ValueError, "unique"):
            validate_training_parity_report(report)

    def test_validator_rejects_non_bool_check_passed(self) -> None:
        report = _report()
        report["checks"][0]["passed"] = 1

        with self.assertRaisesRegex(ValueError, "backend_metadata.passed"):
            validate_training_parity_report(report)


def _report(*, failed_checks: list[str] | None = None) -> dict:
    failed_checks = list(failed_checks or [])
    checks = [
        _check("backend_metadata", failed_checks),
        _check("training_final_loss", failed_checks),
    ]
    return {
        "schema_version": 1,
        "kind": "transformer_training_parity_report",
        "fixture_id": "report-validation-fixture",
        "candidate_backend": "pytorch",
        "passed": not failed_checks,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_check_count": sum(1 for check in checks if check["passed"]),
            "failed_checks": failed_checks,
        },
    }


def _check(name: str, failed_checks: list[str]) -> dict:
    return {"name": name, "passed": name not in failed_checks}


if __name__ == "__main__":
    unittest.main()
