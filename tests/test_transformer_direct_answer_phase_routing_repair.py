import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformer_direct_answer_phase import run_direct_answer_training_loop
from transformer_routing_repair_batch_evidence import ROUTING_REPAIR_BATCH_MODE
from transformer_routing_repair_bundle import PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE


class FakeRng:
    def getstate(self) -> str:
        return "rng-state"


class FakeCursor:
    def __init__(self) -> None:
        self.items = ["a"]
        self.index = 0

    def next(self) -> str:
        item = self.items[self.index]
        self.index += 1
        return item


class FakeModel:
    def to_dict(self, tokenizer: object) -> dict[str, object]:
        return {"tokenizer": tokenizer}


class FakeOptimizer:
    def to_dict(self) -> dict[str, object]:
        return {"optimizer": True}


class TransformerDirectAnswerPhaseRoutingRepairTest(unittest.TestCase):
    def test_routing_repair_bundle_uses_update_search(self) -> None:
        retry_steps: list[int] = []
        batch_steps: list[int] = []

        def record_batch_step(**kwargs: object) -> dict[str, object]:
            batch_steps.append(int(kwargs["direct_step"]))
            return {"step": kwargs["direct_step"], "profiles": ["qa"]}

        def batch_summary(
            args: object,
            steps: list[dict[str, object]],
            baseline: dict[str, object],
        ) -> dict[str, object]:
            return {"passed": True, "steps": steps, "baseline": baseline}

        def retry_update(context: object) -> SimpleNamespace:
            retry_steps.append(int(context.direct_step))
            return SimpleNamespace(loss=0.75, update_guard_applied=True)

        with (
            patch(
                "transformer_direct_answer_phase_loop.record_routing_repair_batch_step",
                side_effect=record_batch_step,
            ),
            patch(
                "transformer_direct_answer_phase_loop.routing_repair_batch_evidence_summary",
                side_effect=batch_summary,
            ),
            patch(
                "transformer_direct_answer_phase_loop.apply_routing_repair_update_search",
                side_effect=retry_update,
            ),
        ):
            result = run_direct_answer_training_loop(
                args=SimpleNamespace(
                    direct_answer_eval_every=0,
                    experiment_bundle=PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
                    direct_answer_mode=ROUTING_REPAIR_BATCH_MODE,
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
                direct_answer_update_guard={},
                direct_baseline={"baseline": True},
                direct_snapshot_recorder=object(),
                best_direct_snapshot=object(),
                last_direct_snapshot={"step": 0},
                last_direct_snapshot_step=0,
                train_adaptive_baseline_floor_update=lambda *args: 0.0,
                train_baseline_anchored_prompt=lambda *args: 0.0,
                restore_direct_update_state=lambda *_: None,
                train_mode_step=lambda **kwargs: SimpleNamespace(
                    loss=1.0,
                    update_guard_applied=False,
                ),
                apply_guard_probe=lambda **kwargs: None,
            )

        self.assertEqual(batch_steps, [1])
        self.assertEqual(retry_steps, [1])
        self.assertEqual(
            result.routing_repair_batch_evidence,
            {
                "passed": True,
                "steps": [{"step": 1, "profiles": ["qa"]}],
                "baseline": {"baseline": True},
            },
        )


if __name__ == "__main__":
    unittest.main()
