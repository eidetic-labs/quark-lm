from __future__ import annotations

import unittest
from unittest.mock import Mock

from transformer_direct_answer_snapshot_lifecycle import (
    finalize_direct_answer_snapshots,
)


class TransformerAnswerSnapshotFinalizationTests(unittest.TestCase):
    def test_appends_final_snapshot_when_last_step_is_behind(self) -> None:
        model = object()
        tokenizer = object()
        optimizer = object()
        recorder = Mock()
        recorder.append.return_value = {"step": 5}
        best_snapshot = Mock()
        best_snapshot.step = 0

        result = finalize_direct_answer_snapshots(
            direct_answer_steps=5,
            restore_best_branch_snapshot=False,
            model_class=Mock(),
            optimizer_class=Mock(),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            recorder=recorder,
            best_snapshot=best_snapshot,
            last_snapshot={"step": 3},
            last_snapshot_step=3,
        )

        recorder.append.assert_called_once_with(5, None)
        best_snapshot.record.assert_called_once_with(
            {"step": 5},
            model,
            tokenizer,
            optimizer,
        )
        self.assertIs(result.model, model)
        self.assertIs(result.tokenizer, tokenizer)
        self.assertIs(result.optimizer, optimizer)
        self.assertEqual(result.last_snapshot, {"step": 5})
        self.assertEqual(result.last_snapshot_step, 5)
        self.assertFalse(result.restored_best_branch_snapshot)

    def test_restores_best_snapshot_when_requested(self) -> None:
        current_model = object()
        current_tokenizer = object()
        current_optimizer = object()
        restored_model = Mock()
        restored_tokenizer = object()
        restored_optimizer = object()
        model_class = Mock()
        model_class.from_dict.return_value = (restored_model, restored_tokenizer)
        optimizer_class = Mock()
        optimizer_class.from_dict.return_value = restored_optimizer
        recorder = Mock()
        recorder.append.return_value = {"step": 5, "restored": True}
        best_snapshot = Mock()
        best_snapshot.step = 2
        best_snapshot.score = (3, 1)
        best_snapshot.model_payload = {"model": True}
        best_snapshot.optimizer_payload = {"optimizer": True}

        result = finalize_direct_answer_snapshots(
            direct_answer_steps=5,
            restore_best_branch_snapshot=True,
            model_class=model_class,
            optimizer_class=optimizer_class,
            model=current_model,
            tokenizer=current_tokenizer,
            optimizer=current_optimizer,
            recorder=recorder,
            best_snapshot=best_snapshot,
            last_snapshot={"step": 5},
            last_snapshot_step=5,
        )

        model_class.from_dict.assert_called_once_with({"model": True})
        optimizer_class.from_dict.assert_called_once_with({"optimizer": True})
        self.assertIs(restored_model.active_optimizer, restored_optimizer)
        self.assertIs(recorder.model(), restored_model)
        self.assertIs(recorder.tokenizer(), restored_tokenizer)
        recorder.append.assert_called_once_with(
            5,
            None,
            {
                "restored_best_branch_snapshot": True,
                "restored_from_step": 2,
                "restored_from_score": [3, 1],
            },
        )
        self.assertIs(result.model, restored_model)
        self.assertIs(result.tokenizer, restored_tokenizer)
        self.assertIs(result.optimizer, restored_optimizer)
        self.assertEqual(result.last_snapshot, {"step": 5, "restored": True})
        self.assertEqual(result.last_snapshot_step, 5)
        self.assertTrue(result.restored_best_branch_snapshot)


if __name__ == "__main__":
    unittest.main()
