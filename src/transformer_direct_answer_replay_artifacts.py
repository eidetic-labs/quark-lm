"""Artifact writing for direct-answer replay plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from corpus_hygiene import attach_replay_plan_summary, write_json_artifact


def write_direct_answer_replay_plan(
    training_plan: dict[str, Any],
    training_plan_path: Path,
    replay_plan: dict[str, Any] | None,
    replay_plan_path: Path | None,
) -> dict[str, Any]:
    if replay_plan_path is None or replay_plan is None:
        return training_plan
    with replay_plan_path.open("w", encoding="utf-8") as handle:
        json.dump(replay_plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    training_plan = attach_replay_plan_summary(training_plan, replay_plan, replay_plan_path)
    write_json_artifact(training_plan_path, training_plan)
    return training_plan
