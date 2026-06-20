from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support.commands import answer_sweep, parse_args
from transformer_answer_sweep_axes import parse_sweep_axes
from transformer_answer_sweep_report import (
    build_answer_sweep_report,
    trial_report_from_metrics,
)


class TransformerAnswerSweepTest(unittest.TestCase):
    def test_parse_sweep_axes_casts_supported_values(self) -> None:
        axes = parse_sweep_axes(
            [
                "tokenizer=char,closed-world-subword",
                "context-size=8,16",
                "learning_rate=0.01",
            ]
        )

        self.assertEqual(axes["tokenizer"], ["char", "closed-world-subword"])
        self.assertEqual(axes["context_size"], [8, 16])
        self.assertEqual(axes["learning_rate"], [0.01])

    def test_answer_sweep_runs_char_and_subword_trials(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "answer-sweep"
            args = parse_args(
                [
                    "answer-sweep",
                    "--run",
                    str(run_dir),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--direct-answer-steps",
                    "0",
                    "--selector-steps",
                    "0",
                    "--generator-steps",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "4",
                    "--feedforward-dim",
                    "8",
                    "--context-size",
                    "8",
                    "--tokenizer-max-token-chars",
                    "4",
                    "--tokenizer-max-new-tokens",
                    "8",
                    "--sweep-axis",
                    "tokenizer=char,closed-world-subword",
                ]
            )

            report = answer_sweep(args)
            saved = json.loads((run_dir / "sweep_report.json").read_text())

            subword_trial = next(
                trial
                for trial in report["trials"]
                if trial["tokenizer_type"] == "closed-world-subword"
            )
            subword_manifest_exists = (
                Path(subword_trial["run"]) / "tokenizer_manifest.json"
            ).exists()

        self.assertEqual(report["kind"], "transformer_answer_sweep_report")
        self.assertEqual(saved["trial_count"], 2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["completed_trials"], 2)
        self.assertEqual(
            set(report["summary"]["tokenizer_types"]),
            {"char", "closed-world-subword"},
        )
        self.assertTrue(subword_manifest_exists)

    def test_trial_report_records_branch_diversity_evidence(self) -> None:
        report = trial_report_from_metrics(
            trial_id="trial-01",
            run_path=Path("runs/trial-01"),
            config={"embedding_dim": 8},
            metrics={
                "constraint_first_promotion": {"status": "rejected"},
                "direct_answer": {
                    "direct_answer_mode": (
                        "branch-profile-balanced-rank-margin-unlikelihood"
                    ),
                    "actual_steps": 4,
                    "direct_answer_update_guard": {
                        "accepted_steps": 2,
                        "rejected_steps": 1,
                        "accepted_learning_rate_scale_counts": {"0.25": 2},
                    },
                    "routing_repair_batch_evidence": {
                        "passed": True,
                        "branch_count": 12,
                        "retention_anchor_count": 0,
                    },
                    "final": {
                        "branch_target_coverage_by_profile": {"qa": 0.25},
                        "branch_diversity_target": {
                            "passed": False,
                            "failed_profiles": 1,
                            "passed_profiles": 0,
                            "max_dominant_predicted_rate": 1.0,
                            "min_target_token_coverage": 0.25,
                            "root_cause": {
                                "mode_counts": {"target_coverage_gap": 1}
                            },
                        },
                    },
                },
            },
        )

        evidence = report["direct_answer_branch_evidence"]
        self.assertEqual(evidence["actual_steps"], 4)
        self.assertEqual(
            evidence["branch_target_coverage_by_profile"],
            {"qa": 0.25},
        )
        self.assertEqual(evidence["update_guard"]["accepted_steps"], 2)
        self.assertEqual(evidence["routing_repair_batch_evidence"]["branch_count"], 12)

    def test_trial_report_compares_branch_evidence_to_frontier(self) -> None:
        report = trial_report_from_metrics(
            trial_id="trial-01",
            run_path=Path("runs/trial-01"),
            config={"embedding_dim": 8},
            metrics=_metrics_with_profile_coverage("trial", {"qa": 0.125}),
            frontier_metrics=_metrics_with_profile_coverage(
                "frontier",
                {"qa": 0.25},
            ),
        )

        comparison = report["frontier_comparison"]
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["frontier_run_id"], "frontier")
        self.assertFalse(comparison["coverage_preserved"])
        self.assertEqual(
            comparison["coverage_diagnostics"]["worst_violation"]["profile"],
            "qa",
        )

    def test_answer_sweep_report_summarizes_frontier_comparisons(self) -> None:
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
        self.assertEqual(report["summary"]["frontier_competitive_trials"], 1)
        self.assertEqual(report["summary"]["frontier_regressed_trials"], 1)

    def test_answer_sweep_backfills_frontier_from_existing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            trial_run = temp_path / "trial-01"
            trial_run.mkdir()
            (trial_run / "transformer_answer_metrics.json").write_text(
                json.dumps(_metrics_with_profile_coverage("trial", {"qa": 0.125})),
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
                json.dumps(_metrics_with_profile_coverage("frontier", {"qa": 0.25})),
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
        self.assertEqual(report["summary"]["frontier_competitive_trials"], 0)
        self.assertEqual(report["summary"]["frontier_regressed_trials"], 1)
        self.assertEqual(saved["report_mode"], "existing_metrics_backfill")


def _metrics_with_profile_coverage(
    run_id: str,
    coverage_by_profile: dict[str, float],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "direct_answer": {
            "final": {
                "branch_profiles": {
                    profile: {
                        "diversity": {
                            "target_unique": 2,
                            "predicted_unique": 1,
                            "target_token_coverage": coverage,
                            "dominant_predicted_rate": 1.0 - coverage,
                            "collapsed": False,
                        },
                        "target_rank": {
                            "top3_rate": coverage,
                            "top5_rate": coverage,
                            "avg": 2.0,
                        },
                    }
                    for profile, coverage in coverage_by_profile.items()
                },
                "branch_target_coverage_by_profile": coverage_by_profile,
                "branch_diversity_target": {
                    "passed": False,
                    "failed_profiles": len(coverage_by_profile),
                    "passed_profiles": 0,
                    "min_target_token_coverage": min(
                        coverage_by_profile.values(),
                    ),
                    "root_cause": {"mode_counts": {"target_coverage_gap": 1}},
                },
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
