from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.transformer_objectives import (
    DIRECT_ANSWER_OBJECTIVE_MODES,
    DirectAnswerObjectiveRegistry,
    PERIODIC_DIRECT_ANSWER_OBJECTIVE_MODES,
    staged_unlikelihood_objective_name,
    validate_direct_answer_objective_mode,
)
from closed_world_lm.transformer_training import (
    JsonlHistoryWriter,
    LossAccumulator,
    ShuffledTrainingCursor,
)


class TransformerTrainingUtilityTests(unittest.TestCase):
    def test_jsonl_history_writer_appends_sorted_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.jsonl"
            writer = JsonlHistoryWriter(path)

            returned = writer.append({"step": 1, "train_loss": 0.5})
            writer.append({"step": 2, "train_loss": 0.25})

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(returned["step"], 1)
        self.assertEqual([row["step"] for row in rows], [1, 2])

    def test_shuffled_training_cursor_cycles_without_empty_pools(self) -> None:
        cursor = ShuffledTrainingCursor(["a", "b", "c"], random.Random(7))
        observed = [cursor.next() for _ in range(6)]

        self.assertEqual(sorted(observed[:3]), ["a", "b", "c"])
        self.assertEqual(sorted(observed[3:]), ["a", "b", "c"])
        with self.assertRaises(ValueError):
            ShuffledTrainingCursor([], random.Random(7))

    def test_loss_accumulator_reports_window_averages(self) -> None:
        accumulator = LossAccumulator()
        accumulator.add(2.0, target_loss=1.0, choice_loss=0.5, choice_candidates=3.0)
        accumulator.add(4.0, target_loss=3.0, choice_loss=1.5, choice_candidates=5.0)

        averages = accumulator.average(2, include_choice=True)
        self.assertEqual(averages["train_loss"], 3.0)
        self.assertEqual(averages["train_target_loss"], 2.0)
        self.assertEqual(averages["train_choice_loss"], 1.0)
        self.assertEqual(averages["train_choice_candidates"], 4.0)
        accumulator.reset()
        self.assertEqual(accumulator.average(1)["train_loss"], 0.0)

    def test_direct_answer_objective_catalog_is_testable(self) -> None:
        self.assertIn("first-error", DIRECT_ANSWER_OBJECTIVE_MODES)
        self.assertIn(
            "periodic-branch-repair-contrast-unlikelihood",
            PERIODIC_DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertIn(
            "branch-context-profile-baseline-floor-stabilization-unlikelihood",
            DIRECT_ANSWER_OBJECTIVE_MODES,
        )
        self.assertEqual(validate_direct_answer_objective_mode("first-error"), "first-error")
        with self.assertRaises(ValueError):
            validate_direct_answer_objective_mode("not-a-mode")

    def test_direct_answer_registry_is_a_small_extension_point(self) -> None:
        registry = DirectAnswerObjectiveRegistry()
        registry.register("demo", lambda **_kwargs: 1.25, "demo rule")

        self.assertEqual(registry.names(), ["demo"])
        self.assertEqual(registry.rules(), {"demo": "demo rule"})
        self.assertEqual(registry.get("demo").train(example="x"), 1.25)

    def test_staged_unlikelihood_names_the_active_subobjective(self) -> None:
        self.assertEqual(
            staged_unlikelihood_objective_name(step=1, total_steps=4),
            "first-error-unlikelihood",
        )
        self.assertEqual(
            staged_unlikelihood_objective_name(step=3, total_steps=4),
            "rollout-unlikelihood",
        )


if __name__ == "__main__":
    unittest.main()
