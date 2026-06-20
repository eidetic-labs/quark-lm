from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.frontier_metrics import metrics_with_profile_coverage
from transformer_direct_answer_frontier_progress import build_frontier_progress_guard
from transformer_direct_answer_snapshot_lifecycle import finalize_direct_answer_snapshots


class TransformerDirectAnswerFrontierProgressTest(unittest.TestCase):
    def test_guard_allows_partial_progress_toward_frontier(self) -> None:
        with _frontier_path(0.5) as frontier_path:
            guard = build_frontier_progress_guard(
                frontier_metrics_path=frontier_path,
                baseline_snapshot=_snapshot(0.0),
                final_snapshot=_snapshot(0.25),
            )

        self.assertTrue(guard["active"])
        self.assertFalse(guard["final_comparison"]["passed"])
        self.assertTrue(guard["score_non_regressed"])
        self.assertTrue(guard["coverage_regression_count_non_increased"])
        self.assertTrue(guard["progress_preserved"])

    def test_guard_rejects_backward_frontier_progress(self) -> None:
        with _frontier_path(0.5) as frontier_path:
            guard = build_frontier_progress_guard(
                frontier_metrics_path=frontier_path,
                baseline_snapshot=_snapshot(0.25),
                final_snapshot=_snapshot(0.0),
            )

        self.assertTrue(guard["active"])
        self.assertFalse(guard["score_non_regressed"])
        self.assertFalse(guard["progress_preserved"])
        self.assertEqual(guard["reason"], "frontier_progress_regressed")

    def test_finalization_restores_baseline_when_frontier_progress_regresses(
        self,
    ) -> None:
        with _frontier_path(0.5) as frontier_path:
            restored_model = SimpleNamespace()
            restored_tokenizer = object()
            restored_optimizer = object()
            model_class = Mock()
            model_class.from_dict.return_value = (restored_model, restored_tokenizer)
            optimizer_class = Mock()
            optimizer_class.from_dict.return_value = restored_optimizer
            recorder = Mock()
            recorder.append.return_value = _snapshot(0.25)

            result = finalize_direct_answer_snapshots(
                direct_answer_steps=1,
                restore_best_branch_snapshot=False,
                model_class=model_class,
                optimizer_class=optimizer_class,
                model=object(),
                tokenizer=object(),
                optimizer=object(),
                recorder=recorder,
                best_snapshot=SimpleNamespace(
                    baseline=_snapshot(0.25),
                    step=0,
                    score=(0.0,),
                    baseline_model_payload={"model": "baseline"},
                    baseline_optimizer_payload={"optimizer": "baseline"},
                ),
                last_snapshot=_snapshot(0.0),
                last_snapshot_step=1,
                frontier_metrics_path=frontier_path,
                frontier_baseline_snapshot=_snapshot(0.25),
            )

        model_class.from_dict.assert_called_once_with({"model": "baseline"})
        optimizer_class.from_dict.assert_called_once_with({"optimizer": "baseline"})
        self.assertIs(restored_model.active_optimizer, restored_optimizer)
        self.assertTrue(result.restored_frontier_progress_snapshot)
        self.assertTrue(result.frontier_progress_guard["progress_preserved"])
        self.assertTrue(result.frontier_progress_guard["pre_restore"]["active"])
        self.assertFalse(
            result.frontier_progress_guard["pre_restore"]["progress_preserved"]
        )


@contextmanager
def _frontier_path(coverage: float) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "frontier_metrics.json"
        path.write_text(
            json.dumps(metrics_with_profile_coverage("frontier", {"qa": coverage})),
            encoding="utf-8",
        )
        yield path


def _snapshot(coverage: float) -> dict[str, object]:
    return {
        "branch_profiles": {
            "qa": {
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
        },
        "branch_target_coverage_by_profile": {"qa": coverage},
        "branch_diversity_target": {
            "passed": False,
            "failed_profiles": 1,
            "passed_profiles": 0,
            "min_target_token_coverage": coverage,
            "root_cause": {"mode_counts": {"target_coverage_gap": 1}},
        },
    }


if __name__ == "__main__":
    unittest.main()
