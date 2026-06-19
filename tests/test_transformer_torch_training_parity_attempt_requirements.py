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


if __name__ == "__main__":
    unittest.main()
