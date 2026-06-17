"""Training recipe artifact lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training_recipe_validation import (
    SCHEMA_VERSION,
    require_non_empty_string,
    require_string_list,
    validate_named_rules,
)


TRAINING_RECIPE_KIND = "training_recipe"


def build_training_recipe(
    version: str,
    component: str,
    run_id: str,
    recipe_id: str,
    purpose: str,
    model: dict[str, Any],
    tokenizer: dict[str, Any],
    data: dict[str, Any],
    objective: dict[str, Any],
    optimizer: dict[str, Any],
    artifacts: list[str | Path],
    gates: list[dict[str, Any]],
    replay: dict[str, Any] | None = None,
    rerun: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRAINING_RECIPE_KIND,
        "version": version,
        "component": component,
        "run_id": run_id,
        "recipe_id": recipe_id,
        "purpose": purpose,
        "uses_external_model": False,
        "model": dict(model),
        "tokenizer": dict(tokenizer),
        "data": dict(data),
        "objective": dict(objective),
        "optimizer": dict(optimizer),
        "replay": dict(replay or {}),
        "artifacts": [str(path) for path in artifacts],
        "gates": [dict(gate) for gate in gates],
        "rerun": dict(rerun or {}),
        "notes": list(notes or []),
    }
    validate_training_recipe(recipe)
    return recipe


def validate_training_recipe(recipe: dict[str, Any]) -> None:
    if recipe.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training recipe schema_version")
    if recipe.get("kind") != TRAINING_RECIPE_KIND:
        raise ValueError(f"kind must be {TRAINING_RECIPE_KIND}")
    for field_name in ("version", "component", "run_id", "recipe_id", "purpose"):
        require_non_empty_string(recipe, field_name)
    if recipe.get("uses_external_model") is not False:
        raise ValueError("training recipes must not use an external model")
    for field_name in (
        "model",
        "tokenizer",
        "data",
        "objective",
        "optimizer",
        "replay",
        "rerun",
    ):
        if not isinstance(recipe.get(field_name), dict):
            raise ValueError(f"{field_name} must be a dict")
    require_string_list(recipe, "artifacts")
    validate_named_rules(recipe.get("gates"), "gates")
    if not isinstance(recipe.get("notes"), list):
        raise ValueError("notes must be a list")


def training_recipe_summary(recipe: dict[str, Any]) -> dict[str, Any]:
    validate_training_recipe(recipe)
    return {
        "recipe_id": recipe["recipe_id"],
        "version": recipe["version"],
        "component": recipe["component"],
        "uses_external_model": False,
        "artifact_count": len(recipe["artifacts"]),
        "gate_count": len(recipe["gates"]),
        "replay_status": recipe["replay"].get("status", "not_declared"),
    }


def write_training_recipe(path: Path, recipe: dict[str, Any]) -> None:
    validate_training_recipe(recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2, sort_keys=True)
        handle.write("\n")


def attach_recipe_summary(
    training_plan: dict[str, Any],
    recipe: dict[str, Any],
    recipe_path: Path,
) -> dict[str, Any]:
    updated = dict(training_plan)
    updated["training_recipe"] = {
        "status": "written",
        "path": str(recipe_path),
        "summary": training_recipe_summary(recipe),
    }
    return updated
