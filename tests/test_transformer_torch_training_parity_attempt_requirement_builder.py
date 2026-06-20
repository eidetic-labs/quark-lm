from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_requirements import (
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
    build_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_parity_attempt_requirement_validation import (
    validate_torch_training_parity_attempt_requirements,
)


class TransformerTorchTrainingParityAttemptRequirementBuilderTests(
    unittest.TestCase
):
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

    def test_runtime_preflight_uses_status_blocker_over_noisy_failures(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={
                "status": "blocked_runtime_unavailable",
                "parity_attempt_allowed": False,
                "summary": {
                    "failed_checks": [
                        "runtime_available",
                        "runtime_kind",
                        "dtype_available",
                    ],
                },
            },
            candidate={},
            report={"passed": False},
        )

        self.assertEqual(requirements["primary_blockers"], ["runtime_available"])
        self.assertEqual(
            requirements["next_actions"],
            ["install_real_pytorch_runtime"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_complete_attempt_is_typed(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={"status": "passed", "parity_attempt_allowed": True},
            candidate={
                "training_readiness": {"status": "ready"},
                "training_replay_parity_gate": {
                    "status": "training_replay_parity_matched",
                    "passed": True,
                },
            },
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

    def test_report_passed_cannot_bypass_runtime_preflight(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={
                "status": "blocked_runtime_unavailable",
                "parity_attempt_allowed": False,
                "summary": {"failed_checks": ["runtime_available"]},
            },
            candidate={
                "training_readiness": {"status": "ready"},
                "training_replay_parity_gate": {
                    "status": "training_replay_parity_matched",
                    "passed": True,
                },
            },
            report={"passed": True},
        )

        self.assertEqual(requirements["stage"], "runtime_preflight")
        self.assertEqual(requirements["status"], "blocked")
        self.assertEqual(
            requirements["next_actions"],
            ["install_real_pytorch_runtime"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_report_passed_cannot_bypass_replay_gate(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={"status": "passed", "parity_attempt_allowed": True},
            candidate={
                "training_readiness": {"status": "ready"},
                "training_replay_parity_gate": {
                    "status": "training_replay_parity_pending",
                    "passed": False,
                    "summary": {"failed_checks": ["replay_buffer"]},
                },
            },
            report={"passed": True},
        )

        self.assertEqual(requirements["stage"], "training_replay_parity")
        self.assertEqual(requirements["status"], "pending")
        self.assertEqual(
            requirements["next_actions"],
            ["resolve_replay_gate:replay_buffer"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_missing_readiness_summary_uses_default_blocker(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={"status": "passed", "parity_attempt_allowed": True},
            candidate={"training_readiness": {"status": "pending"}},
            report={"passed": False},
        )

        self.assertEqual(requirements["stage"], "training_readiness")
        self.assertEqual(requirements["primary_blockers"], ["training_readiness"])
        self.assertEqual(
            requirements["next_actions"],
            ["satisfy_training_readiness:training_readiness"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)


if __name__ == "__main__":
    unittest.main()
