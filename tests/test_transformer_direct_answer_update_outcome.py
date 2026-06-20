from __future__ import annotations

import unittest

from transformer_direct_answer_update_outcome import (
    direct_answer_weight_update_outcome,
)


class TransformerDirectAnswerUpdateOutcomeTest(unittest.TestCase):
    def test_skipped_training_reports_rejected_skip(self) -> None:
        outcome = direct_answer_weight_update_outcome(
            direct_steps_to_run=4,
            training_skipped=True,
            skip_reason="context_gate_failed",
            restored_best_branch_snapshot=False,
            restored_frontier_progress_snapshot=False,
            frontier_progress_guard={"active": False, "progress_preserved": None},
        )

        self.assertEqual(outcome["status"], "skipped")
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["reason"], "context_gate_failed")

    def test_zero_direct_steps_reports_not_run(self) -> None:
        outcome = direct_answer_weight_update_outcome(
            direct_steps_to_run=0,
            training_skipped=False,
            skip_reason=None,
            restored_best_branch_snapshot=False,
            restored_frontier_progress_snapshot=False,
            frontier_progress_guard={"active": False, "progress_preserved": None},
        )

        self.assertEqual(outcome["status"], "not_run")
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["reason"], "no_direct_answer_steps")

    def test_frontier_restore_reports_rejected_update(self) -> None:
        outcome = direct_answer_weight_update_outcome(
            direct_steps_to_run=8,
            training_skipped=False,
            skip_reason=None,
            restored_best_branch_snapshot=False,
            restored_frontier_progress_snapshot=True,
            frontier_progress_guard={
                "active": True,
                "progress_preserved": True,
                "pre_restore": {
                    "progress_preserved": False,
                    "reason": "frontier_progress_regressed",
                },
            },
        )

        self.assertEqual(outcome["status"], "rejected_frontier_restore")
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["reason"], "frontier_progress_regressed")
        self.assertFalse(outcome["pre_restore_progress_preserved"])

    def test_best_snapshot_restore_reports_accepted_restored_update(self) -> None:
        outcome = direct_answer_weight_update_outcome(
            direct_steps_to_run=8,
            training_skipped=False,
            skip_reason=None,
            restored_best_branch_snapshot=True,
            restored_frontier_progress_snapshot=False,
            frontier_progress_guard={"active": True, "progress_preserved": True},
        )

        self.assertEqual(outcome["status"], "accepted_best_snapshot_restore")
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["reason"], "best_branch_snapshot_restored")

    def test_unrestored_direct_steps_report_accepted_update(self) -> None:
        outcome = direct_answer_weight_update_outcome(
            direct_steps_to_run=8,
            training_skipped=False,
            skip_reason=None,
            restored_best_branch_snapshot=False,
            restored_frontier_progress_snapshot=False,
            frontier_progress_guard={"active": True, "progress_preserved": True},
        )

        self.assertEqual(outcome["status"], "accepted")
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["reason"], "frontier_progress_preserved")


if __name__ == "__main__":
    unittest.main()
