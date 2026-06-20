from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_requirement_routing import (
    allowed_torch_training_requirement_statuses,
    torch_training_requirement_action_prefix,
    validate_torch_training_requirement_routing,
)


class TransformerTorchTrainingParityAttemptRequirementRoutingTests(
    unittest.TestCase
):
    def test_exact_routing_accepts_blocker_actions(self) -> None:
        validate_torch_training_requirement_routing(
            stage="training_replay_parity",
            status="pending",
            primary_blockers=["replay_buffer", "replay_update"],
            next_actions=[
                "resolve_replay_gate:replay_buffer",
                "resolve_replay_gate:replay_update",
            ],
            runtime_status="passed",
            exact_actions=True,
            require_blockers=True,
        )

    def test_exact_routing_rejects_blocker_action_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage blockers"):
            validate_torch_training_requirement_routing(
                stage="training_replay_parity",
                status="pending",
                primary_blockers=["replay_buffer"],
                next_actions=["satisfy_training_readiness:replay_buffer"],
                runtime_status="passed",
                exact_actions=True,
                require_blockers=True,
            )

    def test_prefix_routing_accepts_audit_action_without_blockers(self) -> None:
        validate_torch_training_requirement_routing(
            stage="training_parity_report",
            status="pending",
            next_actions=["resolve_training_parity_check:training_final_loss"],
            runtime_status="passed",
            exact_actions=False,
            require_blockers=False,
            action_error="audit.next_actions",
        )

    def test_prefix_routing_rejects_empty_action_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "audit.next_actions"):
            validate_torch_training_requirement_routing(
                stage="training_parity_report",
                status="pending",
                next_actions=["resolve_training_parity_check:"],
                runtime_status="passed",
                exact_actions=False,
                require_blockers=False,
                action_error="audit.next_actions",
            )

    def test_runtime_preflight_routes_from_runtime_status(self) -> None:
        validate_torch_training_requirement_routing(
            stage="runtime_preflight",
            status="blocked",
            primary_blockers=["dtype_available"],
            next_actions=["request_available_pytorch_dtype"],
            runtime_status="blocked_dtype_unavailable",
            exact_actions=True,
            require_blockers=True,
        )

    def test_runtime_preflight_rejects_missing_runtime_blocker(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime blocker"):
            validate_torch_training_requirement_routing(
                stage="runtime_preflight",
                status="blocked",
                primary_blockers=["runtime_available"],
                next_actions=["request_available_pytorch_dtype"],
                runtime_status="blocked_dtype_unavailable",
                exact_actions=True,
                require_blockers=True,
            )

    def test_complete_routing_rejects_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_requirement_routing(
                stage="complete",
                status="satisfied",
                next_actions=["resolve_replay_gate:replay_buffer"],
                runtime_status="passed",
            )

    def test_status_and_prefix_catalogs_are_shared(self) -> None:
        self.assertEqual(
            allowed_torch_training_requirement_statuses("training_readiness"),
            ("blocked", "pending"),
        )
        self.assertEqual(
            torch_training_requirement_action_prefix("training_replay_parity"),
            "resolve_replay_gate",
        )

    def test_unsupported_stage_uses_custom_error_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "audit.next_requirements_stage"):
            validate_torch_training_requirement_routing(
                stage="unknown",
                status="pending",
                next_actions=[],
                runtime_status="passed",
                stage_error="audit.next_requirements_stage",
            )

    def test_unsupported_status_uses_custom_error_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "audit.next_requirements_status"):
            validate_torch_training_requirement_routing(
                stage="complete",
                status="pending",
                next_actions=[],
                runtime_status="passed",
                status_error="audit.next_requirements_status",
            )


if __name__ == "__main__":
    unittest.main()
