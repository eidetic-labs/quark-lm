from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from transformer_direct_answer_update_guard import apply_direct_update_guard_probe


class TransformerDirectAnswerUpdateGuardProbeTests(unittest.TestCase):
    def test_accepts_probe_when_coverage_is_preserved(self) -> None:
        guard = {"checked_steps": 0, "attempted_updates": 0}
        recorder = Mock()
        recorder.record.return_value = {"probe": True}
        restore = Mock()

        with (
            patch(
                "transformer_direct_answer_update_guard."
                "branch_diversity_snapshot_preserves_target_coverage",
                return_value=True,
            ) as preserves,
            patch(
                "transformer_direct_answer_update_guard."
                "record_direct_update_guard_acceptance",
            ) as acceptance,
        ):
            accepted = apply_direct_update_guard_probe(
                direct_answer_update_guard=guard,
                direct_baseline={"baseline": True},
                direct_step=3,
                direct_snapshot_recorder=recorder,
                pre_update_model_payload={"model": True},
                pre_update_optimizer_payload={"optimizer": True},
                restore_direct_update_state=restore,
            )

        self.assertTrue(accepted)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], 1)
        recorder.record.assert_called_once_with(
            3,
            None,
            {
                "baseline_floor_update_guard_probe": True,
                "learning_rate_scale": 1.0,
            },
        )
        preserves.assert_called_once_with({"probe": True}, {"baseline": True})
        acceptance.assert_called_once_with(guard, 1.0)
        restore.assert_not_called()

    def test_rejects_probe_and_restores_when_coverage_regresses(self) -> None:
        guard = {"checked_steps": 0, "attempted_updates": 0, "rejected_steps": 0}
        recorder = Mock()
        recorder.record.return_value = {"probe": True}
        restore = Mock()

        with (
            patch(
                "transformer_direct_answer_update_guard."
                "branch_diversity_snapshot_preserves_target_coverage",
                return_value=False,
            ),
            patch(
                "transformer_direct_answer_update_guard."
                "record_direct_update_guard_rejection_attempt",
            ) as rejection,
        ):
            accepted = apply_direct_update_guard_probe(
                direct_answer_update_guard=guard,
                direct_baseline={"baseline": True},
                direct_step=4,
                direct_snapshot_recorder=recorder,
                pre_update_model_payload={"model": True},
                pre_update_optimizer_payload={"optimizer": True},
                restore_direct_update_state=restore,
            )

        self.assertFalse(accepted)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(guard["attempted_updates"], 1)
        self.assertEqual(guard["rejected_steps"], 1)
        rejection.assert_called_once_with(
            guard,
            {"baseline": True},
            4,
            {"probe": True},
            1.0,
        )
        restore.assert_called_once_with({"model": True}, {"optimizer": True})


if __name__ == "__main__":
    unittest.main()
