from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (  # noqa: E402
    TORCH_TRAINING_REPLAY_GATE_BOOLEAN_CHECKS,
    TORCH_TRAINING_REPLAY_GATE_CHECKS,
    TORCH_TRAINING_REPLAY_GATE_CONTROL_COUNT_CHECK,
    TORCH_TRAINING_REPLAY_GATE_PROBE_CHECKS,
    TORCH_TRAINING_REPLAY_GATE_STATUS_CHECKS,
    validate_torch_training_replay_gate_check,
    validate_torch_training_replay_parity_gate,
)


class TransformerTorchBackendPublicReplayTests(unittest.TestCase):
    def test_replay_gate_contract_is_public(self) -> None:
        self.assertTrue(callable(validate_torch_training_replay_gate_check))
        self.assertTrue(callable(validate_torch_training_replay_parity_gate))
        self.assertIn("replay_buffer", TORCH_TRAINING_REPLAY_GATE_CHECKS)
        self.assertEqual(
            TORCH_TRAINING_REPLAY_GATE_BOOLEAN_CHECKS,
            ("runtime_available", "dtype_available"),
        )
        self.assertEqual(
            TORCH_TRAINING_REPLAY_GATE_STATUS_CHECKS["runtime_kind"],
            "pytorch",
        )
        self.assertEqual(
            TORCH_TRAINING_REPLAY_GATE_CONTROL_COUNT_CHECK,
            "replay_gradient_signatures",
        )
        self.assertIn(
            "replay_final_evaluation",
            TORCH_TRAINING_REPLAY_GATE_PROBE_CHECKS,
        )

    def test_replay_gate_check_validator_is_public(self) -> None:
        check = {
            "name": "replay_buffer",
            "passed": True,
            "probe_passed": True,
            "expected": "replay_buffer_signature_matched",
            "status": "replay_buffer_signature_matched",
            "schema_version": 1,
            "expected_schema_version": 1,
            "proof_flags": {"buffered_gradient_parity_proven": True},
        }

        validate_torch_training_replay_gate_check(check)


if __name__ == "__main__":
    unittest.main()
