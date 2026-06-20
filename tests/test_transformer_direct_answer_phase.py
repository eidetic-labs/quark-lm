import unittest
from types import SimpleNamespace

from transformer_direct_answer_phase import (
    complete_direct_answer_phase,
    run_direct_answer_training_loop,
)


class FakeRng:
    def getstate(self) -> str:
        return "rng-state"


class FakeCursor:
    def __init__(self) -> None:
        self.items = ["a", "b"]
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


class FakeRecorder:
    def __init__(self) -> None:
        self.appended: list[tuple[int, float | None]] = []

    def append(self, step: int, train_loss: float | None) -> dict[str, object]:
        self.appended.append((step, train_loss))
        return {"step": step, "train_loss": train_loss}


class FakeBestSnapshot:
    step = 0
    score = (0.0,)

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record(
        self,
        snapshot: dict[str, object],
        model: object,
        tokenizer: object,
        optimizer: object,
    ) -> None:
        self.records.append(snapshot)


class TransformerDirectAnswerPhaseTest(unittest.TestCase):
    def test_training_loop_records_eval_and_guard_probe(self) -> None:
        guard_calls: list[int] = []
        mode_examples: list[str] = []
        recorder = FakeRecorder()
        best_snapshot = FakeBestSnapshot()

        def train_mode_step(**kwargs: object) -> SimpleNamespace:
            mode_examples.append(str(kwargs["example"]))
            return SimpleNamespace(loss=float(kwargs["direct_step"]), update_guard_applied=False)

        def apply_guard_probe(**kwargs: object) -> None:
            guard_calls.append(int(kwargs["direct_step"]))

        result = run_direct_answer_training_loop(
            args=SimpleNamespace(direct_answer_eval_every=2),
            model=FakeModel(),
            tokenizer=object(),
            optimizer=FakeOptimizer(),
            direct_lessons={"a": "lesson-a", "b": "lesson-b"},
            direct_training_pool=["a", "b"],
            direct_training_cursor=FakeCursor(),
            direct_rng=FakeRng(),
            direct_steps_to_run=2,
            direct_answer_terminator="\n",
            direct_params=["param"],
            direct_answer_baseline_floor_update_gate_active=True,
            direct_answer_baseline_floor_adaptive_updates_active=False,
            direct_answer_update_guard={},
            direct_baseline={"baseline": True},
            direct_snapshot_recorder=recorder,
            best_direct_snapshot=best_snapshot,
            last_direct_snapshot={"step": 0},
            last_direct_snapshot_step=0,
            train_adaptive_baseline_floor_update=lambda *args: 0.0,
            train_baseline_anchored_prompt=lambda *args: 0.0,
            restore_direct_update_state=lambda *_: None,
            train_mode_step=train_mode_step,
            apply_guard_probe=apply_guard_probe,
        )

        self.assertEqual(mode_examples, ["a", "b"])
        self.assertEqual(guard_calls, [1, 2])
        self.assertEqual(recorder.appended, [(2, 1.5)])
        self.assertEqual(best_snapshot.records, [{"step": 2, "train_loss": 1.5}])
        self.assertEqual(result.last_snapshot, {"step": 2, "train_loss": 1.5})
        self.assertEqual(result.last_snapshot_step, 2)

    def test_complete_phase_restores_snapshot_and_builds_metrics(self) -> None:
        direct_setup = SimpleNamespace(
            direct_history_path="history.jsonl",
            direct_profile_aware_targets=["qa"],
            direct_replay_plan_path="replay.json",
            direct_replay_plan={"replay": True},
            direct_replay_prediction_overrides={"override": True},
            direct_replay_prediction_anchors_active=True,
            direct_memory_consolidation_source_plan_path="memory.json",
            direct_memory_consolidation_target_profiles=["memory"],
            direct_memory_consolidation_top_priority_profiles=["top"],
            direct_memory_consolidation_collapsed_memory_backed_profiles=["backed"],
            direct_memory_consolidation_missing_first_token_values=[" answer"],
            direct_memory_consolidation_missing_first_token_ids={"qa": [1]},
            direct_memory_consolidation_profile_specific_missing_first_token_target_map={
                "qa": [1]
            },
        )

        def finalize_snapshots(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                model="restored-model",
                tokenizer="restored-tokenizer",
                optimizer="restored-optimizer",
                last_snapshot={"direct": True},
                restored_best_branch_snapshot=True,
            )

        def build_metrics(**kwargs: object) -> dict[str, object]:
            return {
                "history": kwargs["direct_history_path"],
                "restored": kwargs["direct_answer_restored_best_branch_snapshot"],
                "post_skipped": kwargs["post_direct_candidate_snapshot_skipped"],
            }

        result = complete_direct_answer_phase(
            args=SimpleNamespace(
                direct_answer_steps=2,
                direct_answer_restore_best_branch_snapshot=True,
                skip_post_direct_snapshot=False,
                steps=3,
            ),
            model_class=object,
            optimizer_class=object,
            model=object(),
            tokenizer=object(),
            optimizer=object(),
            direct_snapshot_recorder=object(),
            best_direct_snapshot=SimpleNamespace(step=1, score=(1.0,)),
            last_direct_snapshot={"step": 1},
            last_direct_snapshot_step=1,
            snapshot=lambda step, loss: {"post_step": step, "loss": loss},
            direct_setup=direct_setup,
            direct_steps_to_run=2,
            direct_training_example_count=3,
            direct_answer_update_guard={},
            direct_baseline={"baseline": True},
            direct_answer_training_skipped=False,
            direct_answer_skip_reason=None,
            branch_context_gate={"passed": True},
            generation_config=object(),
            direct_answer_terminator="\n",
            context_coverage={},
            finalize_snapshots=finalize_snapshots,
            build_metrics=build_metrics,
        )

        self.assertEqual(result.model, "restored-model")
        self.assertEqual(result.last_snapshot, {"direct": True})
        self.assertEqual(result.post_direct_candidate_snapshot, {"post_step": 5, "loss": None})
        self.assertEqual(
            result.metrics,
            {"history": "history.jsonl", "restored": True, "post_skipped": False},
        )


if __name__ == "__main__":
    unittest.main()
