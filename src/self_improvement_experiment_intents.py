"""Experiment intent assembly for self-improvement cycles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiment_registry import ExperimentIntent
from self_improvement_experiment_contract import (
    SELF_IMPROVEMENT_COMPONENT,
    SELF_IMPROVEMENT_RECIPE_ID,
    allowed_data_sources,
    experiment_run_id,
    experiment_version,
    planned_experiment_artifacts,
    self_improvement_acceptance_gates,
    self_improvement_failure_criteria,
    self_improvement_hypothesis,
    self_improvement_notes,
)


def self_improvement_experiment_intent(
    args: argparse.Namespace,
    run_dir: Path,
    attempt_dir: Path,
    train_text_path: Path,
) -> dict[str, Any]:
    intent = ExperimentIntent(
        version=experiment_version(args),
        run_id=experiment_run_id(run_dir, attempt_dir),
        component=SELF_IMPROVEMENT_COMPONENT,
        hypothesis=self_improvement_hypothesis(args),
        allowed_data_sources=allowed_data_sources(args, train_text_path),
        planned_artifacts=planned_experiment_artifacts(run_dir, attempt_dir),
        training_recipe_id=SELF_IMPROVEMENT_RECIPE_ID,
        acceptance_gates=self_improvement_acceptance_gates(),
        failure_criteria=self_improvement_failure_criteria(),
        notes=self_improvement_notes(args),
    )
    return intent.to_record()
