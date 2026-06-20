from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support.commands import answer_sweep, parse_args
from support.frontier_metrics import metrics_with_profile_coverage
from transformer_answer_sweep_report import build_answer_sweep_report


class TransformerAnswerSweepFrontierTest(unittest.TestCase):
    def test_report_summary_fails_frontier_regressions(self) -> None:
        frontier_path = Path("runs/frontier/transformer_answer_metrics.json")
        report = build_answer_sweep_report(
            run_id="sweep",
            axes={},
            trials=[
                {
                    "status": "completed",
                    "frontier_comparison": {"available": True, "passed": False},
                },
                {
                    "status": "completed",
                    "frontier_comparison": {"available": True, "passed": True},
                },
                {
                    "status": "completed",
                    "frontier_comparison": {"available": False, "passed": False},
                },
            ],
            max_trials=3,
            dry_run=False,
            frontier_metrics_path=frontier_path,
        )

        self.assertEqual(report["frontier_metrics_path"], str(frontier_path))
        self.assertFalse(report["summary"]["passed"])
        self.assertTrue(report["summary"]["completed_all_trials"])
        self.assertFalse(report["summary"]["frontier_gate_passed"])
        self.assertEqual(report["summary"]["frontier_competitive_trials"], 1)
        self.assertEqual(report["summary"]["frontier_regressed_trials"], 1)
        self.assertEqual(report["summary"]["frontier_reference_active_trials"], 0)
        self.assertEqual(report["summary"]["frontier_reference_training_use_trials"], 0)

    def test_report_summary_passes_without_frontier_regressions(self) -> None:
        report = build_answer_sweep_report(
            run_id="sweep",
            axes={},
            trials=[
                {
                    "status": "completed",
                    "frontier_comparison": {"available": True, "passed": True},
                },
                {"status": "completed"},
            ],
            max_trials=2,
            dry_run=False,
        )

        self.assertTrue(report["summary"]["passed"])
        self.assertTrue(report["summary"]["completed_all_trials"])
        self.assertTrue(report["summary"]["frontier_gate_passed"])

    def test_report_summary_counts_frontier_reference_usage(self) -> None:
        report = build_answer_sweep_report(
            run_id="sweep",
            axes={},
            trials=[
                {
                    "status": "completed",
                    "direct_answer_frontier_reference": {
                        "active": True,
                        "used_for_training": False,
                    },
                },
                {
                    "status": "completed",
                    "direct_answer_frontier_reference": {
                        "active": True,
                        "used_for_training": True,
                    },
                },
            ],
            max_trials=2,
            dry_run=False,
        )

        self.assertEqual(report["summary"]["frontier_reference_active_trials"], 2)
        self.assertEqual(report["summary"]["frontier_reference_training_use_trials"], 1)

    def test_backfills_frontier_from_existing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            trial_run = temp_path / "trial-01"
            trial_run.mkdir()
            (trial_run / "transformer_answer_metrics.json").write_text(
                json.dumps(metrics_with_profile_coverage("trial", {"qa": 0.125})),
                encoding="utf-8",
            )
            source_report = temp_path / "source_sweep_report.json"
            source_report.write_text(
                json.dumps(
                    {
                        "run_id": "source-sweep",
                        "axes": {"embedding_dim": [4]},
                        "max_trials": 1,
                        "trials": [
                            {
                                "trial_id": "trial-01",
                                "status": "completed",
                                "run": str(trial_run),
                                "config": {"embedding_dim": 4},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frontier_path = temp_path / "frontier_metrics.json"
            frontier_path.write_text(
                json.dumps(metrics_with_profile_coverage("frontier", {"qa": 0.25})),
                encoding="utf-8",
            )
            run_dir = temp_path / "rebuilt-sweep"
            args = parse_args(
                [
                    "answer-sweep",
                    "--run",
                    str(run_dir),
                    "--sweep-existing-report",
                    str(source_report),
                    "--sweep-frontier-metrics",
                    str(frontier_path),
                ]
            )

            report = answer_sweep(args)
            saved = json.loads((run_dir / "sweep_report.json").read_text())

        self.assertEqual(report["report_mode"], "existing_metrics_backfill")
        self.assertEqual(report["source_report_path"], str(source_report))
        self.assertFalse(report["summary"]["passed"])
        self.assertEqual(report["summary"]["frontier_competitive_trials"], 0)
        self.assertEqual(report["summary"]["frontier_regressed_trials"], 1)
        self.assertEqual(saved["report_mode"], "existing_metrics_backfill")


if __name__ == "__main__":
    unittest.main()
