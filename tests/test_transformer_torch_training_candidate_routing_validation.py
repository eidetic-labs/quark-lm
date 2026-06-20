from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_TRAINING_CANDIDATE_ROUTE_FIELDS,
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
    TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
    validate_torch_training_candidate_routing,
)


class TransformerTorchTrainingCandidateRoutingValidationTests(unittest.TestCase):
    def test_validator_accepts_missing_runtime_route(self) -> None:
        validate_torch_training_candidate_routing(
            _candidate(
                runtime={"available": False, "dtype_available": False},
                implementation_status="runtime_unavailable",
                parity_status="failed",
                training_case_status="blocked",
            )
        )

    def test_validator_accepts_dtype_unavailable_route(self) -> None:
        validate_torch_training_candidate_routing(
            _candidate(
                runtime={"available": True, "dtype_available": False},
                implementation_status="dtype_unavailable",
                parity_status="pending",
                training_case_status="pending",
            )
        )

    def test_validator_accepts_incomplete_training_runtime_route(self) -> None:
        validate_torch_training_candidate_routing(
            _candidate(
                readiness_status="pending",
                implementation_status=TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
                parity_status="pending",
                training_case_status="pending",
            )
        )

    def test_validator_accepts_pending_replay_route(self) -> None:
        validate_torch_training_candidate_routing(
            _candidate(
                replay_status=TORCH_TRAINING_REPLAY_PENDING_STATUS,
                implementation_status=TORCH_TRAINING_REPLAY_PENDING_STATUS,
                parity_status="pending",
                training_case_status="pending",
            )
        )

    def test_validator_accepts_matched_replay_route(self) -> None:
        validate_torch_training_candidate_routing(
            _candidate(
                replay_status=TORCH_TRAINING_REPLAY_MATCHED_STATUS,
                implementation_status=TORCH_TRAINING_REPLAY_MATCHED_STATUS,
                parity_status="matched",
                training_case_status="matched",
            )
        )

    def test_validator_rejects_stale_implementation_status(self) -> None:
        candidate = _candidate(implementation_status="dtype_unavailable")

        with self.assertRaisesRegex(ValueError, "implementation_status"):
            validate_torch_training_candidate_routing(candidate)

    def test_validator_rejects_stale_backend_parity_status(self) -> None:
        candidate = _candidate(parity_status="failed")

        with self.assertRaisesRegex(ValueError, "backend.parity_status"):
            validate_torch_training_candidate_routing(candidate)

    def test_validator_rejects_stale_training_case_status(self) -> None:
        candidate = _candidate(training_case_status="blocked")

        with self.assertRaisesRegex(ValueError, "training_case.status"):
            validate_torch_training_candidate_routing(candidate)

    def test_route_field_catalog_is_public(self) -> None:
        self.assertEqual(
            TORCH_TRAINING_CANDIDATE_ROUTE_FIELDS,
            (
                "implementation_status",
                "backend.parity_status",
                "training_case.status",
            ),
        )


def _candidate(
    *,
    runtime: dict | None = None,
    readiness_status: str = "ready",
    replay_status: str = TORCH_TRAINING_REPLAY_PENDING_STATUS,
    implementation_status: str = TORCH_TRAINING_REPLAY_PENDING_STATUS,
    parity_status: str = "pending",
    training_case_status: str = "pending",
) -> dict:
    return {
        "implementation_status": implementation_status,
        "runtime": runtime or {"available": True, "dtype_available": True},
        "training_readiness": {"status": readiness_status},
        "training_replay_parity_gate": {"status": replay_status},
        "backend": {"parity_status": parity_status},
        "training_case": {"status": training_case_status},
    }


if __name__ == "__main__":
    unittest.main()
