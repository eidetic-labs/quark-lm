from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (  # noqa: E402
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
    TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS,
    build_torch_training_parity_attempt_requirements,
    validate_torch_training_parity_attempt_requirements,
    validate_torch_training_parity_attempt_summaries,
)


class TransformerTorchBackendPublicRequirementsTests(unittest.TestCase):
    def test_training_attempt_requirements_contract_is_public(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={
                "status": "ready_for_pytorch_parity",
                "passed": True,
                "parity_attempt_allowed": True,
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

        self.assertEqual(
            requirements["kind"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
        )
        self.assertEqual(
            requirements["schema_version"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
        )
        self.assertIn("complete", TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES)
        self.assertIn(
            "install_real_pytorch_runtime",
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS[
                "blocked_runtime_unavailable"
            ],
            "install_real_pytorch_runtime",
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS[
                "blocked_runtime_unavailable"
            ],
            "runtime_available",
        )
        self.assertIn(
            "model_quality_gate",
            TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
        )
        self.assertIn(
            "general_training_backend_gate",
            TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_training_attempt_summary_validation_is_public(self) -> None:
        attempt = {
            "corpus": {
                "corpus_dir": "corpus",
                "train_sha256": "a" * 64,
                "train_chars": 1,
                "manifest": {},
            },
            "runtime": {
                "status": "ready_for_pytorch_parity",
                "passed": True,
                "parity_attempt_allowed": True,
                "failed_checks": [],
                "runtime_kind": "pytorch",
                "device": "cpu",
                "dtype": "float64",
                "runtime_report_sha256": "a" * 64,
            },
            "candidate": {
                "implementation_status": "training_replay_parity_matched",
                "parity_status": "matched",
                "training_readiness_status": "ready",
                "training_readiness_failed_checks": [],
                "training_case_status": "matched",
                "candidate_sha256": "b" * 64,
            },
            "training_replay_parity_gate": {
                "status": "training_replay_parity_matched",
                "passed": True,
                "failed_checks": [],
                "training_replay_parity_gate_sha256": "c" * 64,
            },
            "training_parity_report": {
                "passed": True,
                "failed_checks": [],
                "training_parity_report_sha256": "d" * 64,
            },
        }

        validate_torch_training_parity_attempt_summaries(attempt)


if __name__ == "__main__":
    unittest.main()
