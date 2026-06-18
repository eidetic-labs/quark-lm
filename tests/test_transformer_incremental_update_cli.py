from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support.incremental_training import (
    record,
    train_incremental_candidate,
    write_jsonl_records,
)
from transformer_char_model import main


class TransformerIncrementalUpdateCliTest(unittest.TestCase):
    def test_incremental_update_cli_accepts_candidate_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_metrics, candidate_metrics = train_incremental_candidate(
                root,
                old_repeats=12,
                new_repeats=4,
                candidate_steps=1500,
            )
            accepted_checkpoint = root / "accepted" / "transformer.json"
            report_path = root / "reports" / "guarded_update.json"
            new_probe = write_jsonl_records(
                root / "probes" / "new.jsonl",
                [record("new", "q2:\na:", " ok!")],
            )
            regression_probe = write_jsonl_records(
                root / "probes" / "regression.jsonl",
                [record("old", "q:\na:", " no")],
            )

            exit_code = main(
                [
                    "incremental-update",
                    "--base-checkpoint",
                    str(base_metrics["checkpoint"]),
                    "--candidate-checkpoint",
                    str(candidate_metrics["checkpoint"]),
                    "--accepted-checkpoint",
                    str(accepted_checkpoint),
                    "--report",
                    str(report_path),
                    "--new-lesson-probe",
                    str(new_probe),
                    "--regression-probe",
                    str(regression_probe),
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            accepted_exists = accepted_checkpoint.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["rejection_reasons"], [])
        self.assertTrue(accepted_exists)


if __name__ == "__main__":
    unittest.main()
