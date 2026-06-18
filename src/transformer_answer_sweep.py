"""Controlled sweep runner for transformer answer-training screens."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from transformer_answer_sweep_axes import build_sweep_trials, parse_sweep_axes
from transformer_answer_sweep_report import (
    build_answer_sweep_report,
    planned_trial_report,
    trial_report_from_metrics,
    write_answer_sweep_report,
)


def run_transformer_answer_sweep(
    args: argparse.Namespace,
    train_answer_trial: Callable[[argparse.Namespace], dict[str, Any]],
) -> dict[str, Any]:
    axes = parse_sweep_axes(args.sweep_axis)
    trials = build_sweep_trials(args, axes)
    if len(trials) > args.sweep_max_trials:
        raise ValueError(
            f"sweep expands to {len(trials)} trials, "
            f"above --sweep-max-trials={args.sweep_max_trials}"
        )
    args.run.mkdir(parents=True, exist_ok=True)
    trial_reports = []
    for trial in trials:
        if args.sweep_dry_run:
            trial_reports.append(
                planned_trial_report(
                    trial_id=trial.trial_id,
                    run_path=trial.args.run,
                    config=trial.config,
                )
            )
            continue
        metrics = train_answer_trial(trial.args)
        trial_reports.append(
            trial_report_from_metrics(
                trial_id=trial.trial_id,
                run_path=trial.args.run,
                config=trial.config,
                metrics=metrics,
            )
        )
    report = build_answer_sweep_report(
        run_id=args.run.name,
        axes=axes,
        trials=trial_reports,
        max_trials=args.sweep_max_trials,
        dry_run=args.sweep_dry_run,
    )
    report_path = args.sweep_report or (args.run / "sweep_report.json")
    report["report_path"] = str(report_path)
    write_answer_sweep_report(report_path, report)
    return report
