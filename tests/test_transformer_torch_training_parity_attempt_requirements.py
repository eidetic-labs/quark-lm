from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_requirements import (
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
    build_torch_training_parity_attempt_requirements,
    validate_torch_training_parity_attempt_requirements,
)


class TransformerTorchTrainingParityAttemptRequirementsTests(unittest.TestCase):
    def test_runtime_preflight_blocker_is_typed(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={
                "status": "blocked_runtime_unavailable",
                "parity_attempt_allowed": False,
                "summary": {"failed_checks": ["runtime_available"]},
            },
            candidate={},
            report={"passed": False},
        )

        self.assertEqual(
            requirements["kind"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
        )
        self.assertEqual(
            requirements["schema_version"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
        )
        self.assertEqual(requirements["stage"], "runtime_preflight")
        self.assertEqual(requirements["status"], "blocked")
        self.assertEqual(
            requirements["next_actions"],
            ["install_real_pytorch_runtime"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_complete_attempt_is_typed(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={"status": "passed", "parity_attempt_allowed": True},
            candidate={"training_readiness": {"status": "ready"}},
            report={"passed": True},
        )

        self.assertEqual(
            requirements["kind"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
        )
        self.assertEqual(requirements["stage"], "complete")
        self.assertEqual(requirements["status"], "satisfied")
        self.assertEqual(requirements["primary_blockers"], [])
        self.assertEqual(requirements["next_actions"], [])
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_unsupported_stage(self) -> None:
        requirements = _valid_requirements()
        requirements["stage"] = "unknown"

        with self.assertRaisesRegex(ValueError, "stage"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_wrong_status_for_stage(self) -> None:
        requirements = _valid_requirements()
        requirements["stage"] = "complete"
        requirements["status"] = "pending"

        with self.assertRaisesRegex(ValueError, "status"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_non_string_actions(self) -> None:
        requirements = _valid_requirements()
        requirements["next_actions"] = [42]

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_missing_reference_field(self) -> None:
        requirements = _valid_requirements()
        requirements.pop("runtime_status")

        with self.assertRaisesRegex(ValueError, "runtime_status"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_exposes_known_stage_catalog(self) -> None:
        self.assertIn(
            "runtime_preflight",
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
        )
        self.assertIn("complete", TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES)


def _valid_requirements() -> dict:
    return build_torch_training_parity_attempt_requirements(
        runtime_report={
            "status": "blocked_dtype_unavailable",
            "parity_attempt_allowed": False,
            "summary": {"failed_checks": ["dtype_available"]},
        },
        candidate={},
        report={"passed": False},
    )


if __name__ == "__main__":
    unittest.main()
