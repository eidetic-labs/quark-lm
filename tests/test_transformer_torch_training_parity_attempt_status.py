from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_status import (
    TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_READY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS,
    resolve_torch_training_parity_attempt_passed,
    resolve_torch_training_parity_attempt_status,
)


class TransformerTorchTrainingParityAttemptStatusTests(unittest.TestCase):
    def test_runtime_blocker_controls_status_before_report(self) -> None:
        status = _status(
            runtime={"status": "blocked_runtime_unavailable"},
            gate={"status": "training_replay_parity_matched", "passed": True},
            report={"passed": True},
        )

        self.assertEqual(status, "blocked_runtime_unavailable")
        self.assertFalse(
            _passed(
                runtime={"status": "blocked_runtime_unavailable"},
                gate={"status": "training_replay_parity_matched", "passed": True},
                report={"passed": True},
            )
        )

    def test_runtime_blocker_uses_fallback_status(self) -> None:
        self.assertEqual(
            _status(
                runtime={},
                gate={"status": "training_replay_parity_matched", "passed": True},
                report={"passed": True},
            ),
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS,
        )

    def test_replay_gate_controls_status_before_report(self) -> None:
        self.assertEqual(
            _status(
                runtime=_ready_runtime(),
                gate={"status": "training_replay_parity_pending", "passed": False},
                report={"passed": True},
            ),
            "training_replay_parity_pending",
        )
        self.assertFalse(
            _passed(
                runtime=_ready_runtime(),
                gate={"status": "training_replay_parity_pending", "passed": False},
                report={"passed": True},
            )
        )

    def test_report_failure_remains_pending_after_replay_match(self) -> None:
        self.assertEqual(
            _status(
                runtime=_ready_runtime(),
                gate={"status": "training_replay_parity_matched", "passed": True},
                report={"passed": False},
            ),
            "training_replay_parity_matched",
        )
        self.assertFalse(
            _passed(
                runtime=_ready_runtime(),
                gate={"status": "training_replay_parity_matched", "passed": True},
                report={"passed": False},
            )
        )

    def test_all_prerequisites_match(self) -> None:
        runtime = _ready_runtime()
        gate = {"status": "training_replay_parity_matched", "passed": True}
        report = {"passed": True}

        self.assertEqual(
            _status(runtime=runtime, gate=gate, report=report),
            TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS,
        )
        self.assertTrue(_passed(runtime=runtime, gate=gate, report=report))

    def test_runtime_status_must_be_ready_before_match(self) -> None:
        runtime = {
            "status": "blocked_test_double_runtime",
            "passed": True,
            "parity_attempt_allowed": True,
        }
        gate = {"status": "training_replay_parity_matched", "passed": True}
        report = {"passed": True}

        self.assertEqual(
            _status(runtime=runtime, gate=gate, report=report),
            "blocked_test_double_runtime",
        )
        self.assertFalse(_passed(runtime=runtime, gate=gate, report=report))

    def test_replay_status_must_be_matched_before_match(self) -> None:
        gate = {"status": "training_replay_parity_pending", "passed": True}
        report = {"passed": True}

        self.assertEqual(
            _status(runtime=_ready_runtime(), gate=gate, report=report),
            "training_replay_parity_pending",
        )
        self.assertFalse(_passed(runtime=_ready_runtime(), gate=gate, report=report))

    def test_resolver_matches_built_attempt_summary(self) -> None:
        attempt = build_torch_training_parity_attempt(
            corpus_dir=ROOT / "corpus",
            fixture_id="attempt-status-resolver",
            seed=53,
            context_index=4,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            steps=2,
            importer=_missing_importer,
        )["attempt"]

        self.assertEqual(
            attempt["status"],
            resolve_torch_training_parity_attempt_status(
                runtime=attempt["runtime"],
                training_replay_parity_gate=attempt["training_replay_parity_gate"],
                training_parity_report=attempt["training_parity_report"],
            ),
        )
        self.assertEqual(
            attempt["passed"],
            resolve_torch_training_parity_attempt_passed(
                runtime=attempt["runtime"],
                training_replay_parity_gate=attempt["training_replay_parity_gate"],
                training_parity_report=attempt["training_parity_report"],
            ),
        )


def _status(
    *,
    runtime: dict,
    gate: dict,
    report: dict,
) -> str:
    return resolve_torch_training_parity_attempt_status(
        runtime=runtime,
        training_replay_parity_gate=gate,
        training_parity_report=report,
    )


def _passed(
    *,
    runtime: dict,
    gate: dict,
    report: dict,
) -> bool:
    return resolve_torch_training_parity_attempt_passed(
        runtime=runtime,
        training_replay_parity_gate=gate,
        training_parity_report=report,
    )


def _ready_runtime() -> dict:
    return {
        "status": TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_READY_STATUS,
        "passed": True,
        "parity_attempt_allowed": True,
    }


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


if __name__ == "__main__":
    unittest.main()
