from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from support.frontier_metrics import metrics_with_profile_coverage
from support.routing_repair_update_search import (
    routing_repair_guard,
    routing_repair_snapshot,
)
from transformer_direct_answer_phase import run_direct_answer_training_loop


class TransformerDirectAnswerFrontierUpdateLoopTest(unittest.TestCase):
    def test_loop_restores_frontier_regressive_direct_update(self) -> None:
        guard = routing_repair_guard()
        restores: list[tuple[dict[str, object], dict[str, object]]] = []
        train_examples: list[str] = []

        def train_mode_step(**kwargs: object) -> SimpleNamespace:
            train_examples.append(str(kwargs["example"]))
            return SimpleNamespace(loss=1.0, update_guard_applied=False)

        def restore_state(
            model_payload: dict[str, object],
            optimizer_payload: dict[str, object],
        ) -> tuple[FakeModel, object, FakeOptimizer, list[str]]:
            restores.append((model_payload, optimizer_payload))
            return FakeModel(), object(), FakeOptimizer(), ["param"]

        with tempfile.TemporaryDirectory() as temp:
            frontier_path = Path(temp) / "frontier_metrics.json"
            frontier_path.write_text(
                json.dumps(metrics_with_profile_coverage("frontier", {"qa": 0.5})),
                encoding="utf-8",
            )

            result = run_direct_answer_training_loop(
                args=SimpleNamespace(
                    direct_answer_eval_every=0,
                    direct_answer_frontier_metrics=frontier_path,
                ),
                model=FakeModel(),
                tokenizer=object(),
                optimizer=FakeOptimizer(),
                direct_lessons={"a": "lesson-a"},
                direct_training_pool=["a"],
                direct_training_cursor=FakeCursor(),
                direct_rng=FakeRng(),
                direct_steps_to_run=1,
                direct_answer_terminator="\n",
                direct_params=["param"],
                direct_answer_baseline_floor_update_gate_active=False,
                direct_answer_baseline_floor_adaptive_updates_active=False,
                direct_answer_update_guard=guard,
                direct_baseline=routing_repair_snapshot(0.5),
                direct_snapshot_recorder=FakeRecorder([routing_repair_snapshot(0.0)]),
                best_direct_snapshot=FakeBestSnapshot(),
                last_direct_snapshot={"step": 0},
                last_direct_snapshot_step=0,
                train_adaptive_baseline_floor_update=lambda *args: 0.0,
                train_baseline_anchored_prompt=lambda *args: 0.0,
                restore_direct_update_state=restore_state,
                train_mode_step=train_mode_step,
            )

        self.assertEqual(train_examples, ["a"])
        self.assertEqual(len(restores), 1)
        self.assertEqual(result.last_snapshot, {"step": 0})
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], 1)
        self.assertEqual(guard["rejected_steps"], 1)
        self.assertEqual(guard["frontier_update_guard_rejections"], 1)


class FakeRng:
    def getstate(self) -> str:
        return "rng-state"


class FakeCursor:
    def next(self) -> str:
        return "a"


class FakeModel:
    def to_dict(self, tokenizer: object) -> dict[str, object]:
        return {"tokenizer": tokenizer}


class FakeOptimizer:
    def to_dict(self) -> dict[str, object]:
        return {"optimizer": True}


class FakeRecorder:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = list(records)

    def record(
        self,
        step: int,
        train_loss: float | None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return self.records.pop(0)


class FakeBestSnapshot:
    step = 0
    score = (0.0,)


if __name__ == "__main__":
    unittest.main()
