from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support.frontier_metrics import metrics_with_profile_coverage
from support.routing_repair_update_search import (
    routing_repair_guard,
    routing_repair_snapshot,
)
from transformer_direct_answer_frontier_update_guard import (
    apply_direct_frontier_update_guard_probe,
)


class TransformerDirectAnswerFrontierUpdateGuardTest(unittest.TestCase):
    def test_accepts_update_that_preserves_frontier_progress(self) -> None:
        guard = routing_repair_guard()
        restores: list[tuple[dict, dict]] = []

        with _frontier_path(0.5) as frontier_path:
            accepted = apply_direct_frontier_update_guard_probe(
                direct_answer_update_guard=guard,
                direct_baseline=routing_repair_snapshot(0.5),
                direct_step=1,
                direct_snapshot_recorder=_Recorder([routing_repair_snapshot(0.75)]),
                frontier_metrics_path=frontier_path,
                pre_update_model_payload={"model": True},
                pre_update_optimizer_payload={"optimizer": True},
                restore_direct_update_state=lambda model, optimizer: restores.append(
                    (model, optimizer)
                ),
            )

        self.assertTrue(accepted)
        self.assertEqual(restores, [])
        self.assertEqual(guard["accepted_steps"], 1)
        self.assertEqual(guard["frontier_update_guard_acceptances"], 1)
        self.assertTrue(guard["frontier_update_guard_last"]["progress_preserved"])

    def test_rejects_and_restores_update_that_regresses_frontier_progress(self) -> None:
        guard = routing_repair_guard()
        restores: list[tuple[dict, dict]] = []

        with _frontier_path(0.5) as frontier_path:
            accepted = apply_direct_frontier_update_guard_probe(
                direct_answer_update_guard=guard,
                direct_baseline=routing_repair_snapshot(0.5),
                direct_step=1,
                direct_snapshot_recorder=_Recorder([routing_repair_snapshot(0.0)]),
                frontier_metrics_path=frontier_path,
                pre_update_model_payload={"model": True},
                pre_update_optimizer_payload={"optimizer": True},
                restore_direct_update_state=lambda model, optimizer: restores.append(
                    (model, optimizer)
                ),
            )

        self.assertFalse(accepted)
        self.assertEqual(restores, [({"model": True}, {"optimizer": True})])
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertEqual(guard["frontier_update_guard_rejections"], 1)
        self.assertEqual(
            guard["frontier_update_guard_last"]["reason"],
            "frontier_progress_regressed",
        )


class _Recorder:
    def __init__(self, snapshots: list[dict]) -> None:
        self.snapshots = snapshots
        self.index = 0

    def record(
        self,
        step: int,
        train_loss: float | None,
        extra: dict | None = None,
    ) -> dict:
        snapshot = self.snapshots[self.index]
        self.index += 1
        return snapshot


class _frontier_path:
    def __init__(self, coverage: float) -> None:
        self.coverage = coverage
        self.temp: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "frontier_metrics.json"
        self.path.write_text(
            json.dumps(metrics_with_profile_coverage("frontier", {"qa": self.coverage})),
            encoding="utf-8",
        )
        return self.path

    def __exit__(self, *args: object) -> None:
        if self.temp is not None:
            self.temp.cleanup()


if __name__ == "__main__":
    unittest.main()
