from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_requirements import (
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
    build_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_parity_attempt_requirement_validation import (
    validate_torch_training_parity_attempt_requirements,
)


class TransformerTorchTrainingParityAttemptRequirementsTests(unittest.TestCase):
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

    def test_validator_rejects_empty_pending_blockers(self) -> None:
        requirements = _valid_requirements()
        requirements["primary_blockers"] = []

        with self.assertRaisesRegex(ValueError, "primary_blockers"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_unsupported_runtime_action(self) -> None:
        requirements = _valid_requirements()
        requirements["next_actions"] = ["resolve_replay_gate:runtime_available"]

        with self.assertRaisesRegex(ValueError, "runtime action"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_runtime_action_status_mismatch(self) -> None:
        requirements = _valid_requirements()
        requirements["next_actions"] = ["install_real_pytorch_runtime"]

        with self.assertRaisesRegex(ValueError, "runtime action"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_runtime_blocker_status_mismatch(self) -> None:
        requirements = _valid_requirements()
        requirements["primary_blockers"] = ["runtime_available"]

        with self.assertRaisesRegex(ValueError, "runtime blocker"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_extra_runtime_blockers(self) -> None:
        requirements = _valid_requirements()
        requirements["primary_blockers"] = [
            "dtype_available",
            "runtime_available",
        ]

        with self.assertRaisesRegex(ValueError, "runtime blocker"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_rejects_stage_action_mismatch(self) -> None:
        requirements = _stage_requirements(
            stage="training_replay_parity",
            blockers=["replay_buffer"],
            actions=["satisfy_training_readiness:replay_buffer"],
        )

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_accepts_stage_prefixed_actions(self) -> None:
        requirements = _stage_requirements(
            stage="training_replay_parity",
            blockers=["replay_buffer", "replay_update"],
            actions=[
                "resolve_replay_gate:replay_buffer",
                "resolve_replay_gate:replay_update",
            ],
        )

        validate_torch_training_parity_attempt_requirements(requirements)

    def test_validator_exposes_known_stage_catalog(self) -> None:
        self.assertIn(
            "runtime_preflight",
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
        )
        self.assertIn("complete", TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES)
        self.assertIn(
            "install_real_pytorch_runtime",
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS[
                "blocked_dtype_unavailable"
            ],
            "request_available_pytorch_dtype",
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS[
                "blocked_dtype_unavailable"
            ],
            "dtype_available",
        )


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


def _stage_requirements(
    *,
    stage: str,
    blockers: list[str],
    actions: list[str],
) -> dict:
    return {
        "schema_version": TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
        "kind": TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
        "stage": stage,
        "status": "pending",
        "primary_blockers": blockers,
        "next_actions": actions,
        "runtime_status": "passed",
        "parity_attempt_allowed": True,
        "training_readiness_status": "ready",
        "training_replay_parity_status": "training_replay_parity_pending",
        "training_report_passed": False,
    }


if __name__ == "__main__":
    unittest.main()
