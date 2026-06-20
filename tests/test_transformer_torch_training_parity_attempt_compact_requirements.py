from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_compact_requirements import (
    build_torch_training_parity_attempt_compact_requirements,
)
from transformer_torch_training_parity_attempt_requirement_validation import (
    validate_torch_training_parity_attempt_requirements,
)


class TransformerTorchTrainingParityAttemptCompactRequirementsTests(
    unittest.TestCase
):
    def test_compact_runtime_preflight_uses_runtime_failed_checks(self) -> None:
        requirements = _requirements(
            runtime={
                "status": "blocked_runtime_unavailable",
                "parity_attempt_allowed": False,
                "failed_checks": ["runtime_available"],
            },
        )

        self.assertEqual(requirements["stage"], "runtime_preflight")
        self.assertEqual(requirements["primary_blockers"], ["runtime_available"])
        self.assertEqual(
            requirements["next_actions"],
            ["install_real_pytorch_runtime"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_compact_readiness_uses_readiness_failed_checks(self) -> None:
        requirements = _requirements(
            runtime=_ready_runtime(),
            candidate={
                "training_readiness_status": "pending",
                "training_readiness_failed_checks": ["adamw_optimizer"],
            },
        )

        self.assertEqual(requirements["stage"], "training_readiness")
        self.assertEqual(requirements["primary_blockers"], ["adamw_optimizer"])
        self.assertEqual(
            requirements["next_actions"],
            ["satisfy_training_readiness:adamw_optimizer"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_compact_replay_gate_uses_replay_failed_checks(self) -> None:
        requirements = _requirements(
            runtime=_ready_runtime(),
            candidate={"training_readiness_status": "ready"},
            gate={
                "status": "training_replay_parity_pending",
                "passed": False,
                "failed_checks": ["replay_buffer"],
            },
            report={"passed": True},
        )

        self.assertEqual(requirements["stage"], "training_replay_parity")
        self.assertEqual(requirements["primary_blockers"], ["replay_buffer"])
        self.assertEqual(
            requirements["next_actions"],
            ["resolve_replay_gate:replay_buffer"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_compact_report_failure_uses_report_failed_checks(self) -> None:
        requirements = _requirements(
            runtime=_ready_runtime(),
            candidate={"training_readiness_status": "ready"},
            gate={
                "status": "training_replay_parity_matched",
                "passed": True,
            },
            report={"passed": False, "failed_checks": ["training_final_loss"]},
        )

        self.assertEqual(requirements["stage"], "training_parity_report")
        self.assertEqual(requirements["primary_blockers"], ["training_final_loss"])
        self.assertEqual(
            requirements["next_actions"],
            ["resolve_training_parity_check:training_final_loss"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_compact_complete_has_no_actions(self) -> None:
        requirements = _requirements(
            runtime=_ready_runtime(),
            candidate={"training_readiness_status": "ready"},
            gate={
                "status": "training_replay_parity_matched",
                "passed": True,
            },
            report={"passed": True},
        )

        self.assertEqual(requirements["stage"], "complete")
        self.assertEqual(requirements["status"], "satisfied")
        self.assertEqual(requirements["primary_blockers"], [])
        self.assertEqual(requirements["next_actions"], [])
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_compact_complete_requires_ready_runtime_status(self) -> None:
        requirements = _requirements(
            runtime={
                "status": "blocked_test_double_runtime",
                "passed": True,
                "parity_attempt_allowed": True,
                "failed_checks": ["runtime_kind"],
            },
            candidate={"training_readiness_status": "ready"},
            gate={
                "status": "training_replay_parity_matched",
                "passed": True,
            },
            report={"passed": True},
        )

        self.assertEqual(requirements["stage"], "runtime_preflight")
        self.assertEqual(requirements["primary_blockers"], ["runtime_kind"])
        self.assertEqual(
            requirements["next_actions"],
            ["run_again_with_real_pytorch_runtime"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_compact_complete_requires_matched_replay_status(self) -> None:
        requirements = _requirements(
            runtime=_ready_runtime(),
            candidate={"training_readiness_status": "ready"},
            gate={
                "status": "training_replay_parity_pending",
                "passed": True,
            },
            report={"passed": True},
        )

        self.assertEqual(requirements["stage"], "training_replay_parity")
        self.assertEqual(
            requirements["primary_blockers"],
            ["training_replay_parity_gate"],
        )
        self.assertEqual(
            requirements["next_actions"],
            ["resolve_replay_gate:training_replay_parity_gate"],
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_compact_rebuild_matches_generated_attempt(self) -> None:
        attempt = build_torch_training_parity_attempt(
            corpus_dir=ROOT / "corpus",
            fixture_id="compact-requirements-training-parity-attempt",
            seed=53,
            context_index=4,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            steps=2,
            importer=_missing_importer,
        )["attempt"]

        self.assertEqual(attempt["next_requirements"], _from_attempt(attempt))


def _requirements(
    *,
    runtime: dict,
    candidate: dict | None = None,
    gate: dict | None = None,
    report: dict | None = None,
) -> dict:
    return build_torch_training_parity_attempt_compact_requirements(
        runtime={
            "status": runtime.get("status"),
            "passed": runtime.get("passed"),
            "parity_attempt_allowed": runtime.get("parity_attempt_allowed", False),
            "failed_checks": list(runtime.get("failed_checks", [])),
        },
        candidate={
            "training_readiness_status": "blocked",
            "training_readiness_failed_checks": [],
            **copy.deepcopy(candidate or {}),
        },
        training_replay_parity_gate={
            "status": "training_replay_parity_pending",
            "passed": False,
            "failed_checks": [],
            **copy.deepcopy(gate or {}),
        },
        training_parity_report={
            "passed": False,
            "failed_checks": [],
            **copy.deepcopy(report or {}),
        },
    )


def _from_attempt(attempt: dict) -> dict:
    return build_torch_training_parity_attempt_compact_requirements(
        runtime=attempt["runtime"],
        candidate=attempt["candidate"],
        training_replay_parity_gate=attempt["training_replay_parity_gate"],
        training_parity_report=attempt["training_parity_report"],
    )


def _ready_runtime() -> dict:
    return {
        "status": "ready_for_pytorch_parity",
        "passed": True,
        "parity_attempt_allowed": True,
    }


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


if __name__ == "__main__":
    unittest.main()
