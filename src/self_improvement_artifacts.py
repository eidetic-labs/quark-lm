"""Artifact writing helpers for self-improvement attempts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from candidate_quarantine import write_candidate_quarantine
from closed_world_verifier_reports import write_verifier_report
from corpus_hygiene import write_json_artifact
from curriculum import write_json
from experiment_registry import write_experiment_intent
from constraint_first_report import write_constraint_first_report
from training_recipe_core import write_training_recipe


ATTEMPT_DIR_RE = re.compile(r"^attempt-(?P<number>\d+)$")


def next_attempt(run_dir: Path) -> tuple[int, Path]:
    attempts_dir = run_dir / "attempts"
    numbers: list[int] = []
    if attempts_dir.exists():
        for child in attempts_dir.iterdir():
            match = ATTEMPT_DIR_RE.match(child.name)
            if match and child.is_dir():
                numbers.append(int(match["number"]))
    number = max(numbers, default=0) + 1
    return number, attempts_dir / f"attempt-{number:03d}"


def write_report_artifacts(
    report: dict[str, Any],
    run_dir: Path,
    attempt_dir: Path,
    attempt_number: int,
) -> None:
    report["attempt"] = {
        "index": attempt_number,
        "path": str(attempt_dir),
        "report": str(attempt_dir / "self_improvement_report.json"),
        "latest_report": str(run_dir / "self_improvement_report.json"),
    }
    write_json(attempt_dir / "corpus_snapshot.json", report["corpus_snapshot"])
    write_json(attempt_dir / "corpus_diff.json", report["corpus_diff"])
    if "corpus_hygiene" in report:
        write_json_artifact(attempt_dir / "corpus_hygiene.json", report["corpus_hygiene"])
    if "training_plan" in report:
        write_json_artifact(attempt_dir / "training_plan.json", report["training_plan"])
    if "training_recipe" in report:
        write_training_recipe(
            attempt_dir / "training_recipe.json",
            report["training_recipe"],
        )
    if "candidate_quarantine" in report:
        write_candidate_quarantine(
            attempt_dir / "candidate_quarantine.json",
            report["candidate_quarantine"],
        )
    if "tokenizer_candidate" in report:
        write_json_artifact(
            attempt_dir / "tokenizer_manifest.json",
            report["tokenizer_candidate"]["manifest"],
        )
        write_json_artifact(
            attempt_dir / "tokenizer_report.json",
            report["tokenizer_candidate"]["report"],
        )
    if "closed_world_verifier" in report:
        write_verifier_report(
            attempt_dir / "closed_world_verifier.json",
            report["closed_world_verifier"],
        )
    if "constraint_first_promotion" in report:
        write_constraint_first_report(
            attempt_dir / "constraint_first_promotion.json",
            report["constraint_first_promotion"],
        )
    if "experiment_intent" in report:
        write_experiment_intent(
            attempt_dir / "experiment_intent.json",
            report["experiment_intent"],
        )
    write_json(attempt_dir / "self_improvement_report.json", report)
    write_json(run_dir / "corpus_snapshot.json", report["corpus_snapshot"])
    write_json(run_dir / "corpus_diff.json", report["corpus_diff"])
    if "corpus_hygiene" in report:
        write_json_artifact(run_dir / "corpus_hygiene.json", report["corpus_hygiene"])
    if "training_plan" in report:
        write_json_artifact(run_dir / "training_plan.json", report["training_plan"])
    if "training_recipe" in report:
        write_training_recipe(
            run_dir / "training_recipe.json",
            report["training_recipe"],
        )
    if "candidate_quarantine" in report:
        write_candidate_quarantine(
            run_dir / "candidate_quarantine.json",
            report["candidate_quarantine"],
        )
    if "tokenizer_candidate" in report:
        write_json_artifact(
            run_dir / "tokenizer_manifest.json",
            report["tokenizer_candidate"]["manifest"],
        )
        write_json_artifact(
            run_dir / "tokenizer_report.json",
            report["tokenizer_candidate"]["report"],
        )
    if "closed_world_verifier" in report:
        write_verifier_report(
            run_dir / "closed_world_verifier.json",
            report["closed_world_verifier"],
        )
    if "constraint_first_promotion" in report:
        write_constraint_first_report(
            run_dir / "constraint_first_promotion.json",
            report["constraint_first_promotion"],
        )
    if "experiment_intent" in report:
        write_experiment_intent(
            run_dir / "experiment_intent.json",
            report["experiment_intent"],
        )
    write_json(run_dir / "self_improvement_report.json", report)
