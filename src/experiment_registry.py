"""Experiment intent records for closed-world training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_INTENT_SCHEMA_VERSION = 1
DECISION_STATUSES = {"planned", "promoted", "rejected", "aborted"}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ExperimentIntent:
    version: str
    run_id: str
    component: str
    hypothesis: str
    allowed_data_sources: list[str]
    planned_artifacts: list[str]
    training_recipe_id: str
    acceptance_gates: list[dict[str, Any]]
    failure_criteria: list[str]
    replay_plan_id: str | None = None
    created_at: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record = {
            "schema_version": EXPERIMENT_INTENT_SCHEMA_VERSION,
            "kind": "experiment_intent",
            "created_at": self.created_at or utc_now_iso(),
            "version": self.version,
            "run_id": self.run_id,
            "component": self.component,
            "hypothesis": self.hypothesis,
            "allowed_data_sources": list(self.allowed_data_sources),
            "planned_artifacts": list(self.planned_artifacts),
            "training_recipe_id": self.training_recipe_id,
            "acceptance_gates": [dict(gate) for gate in self.acceptance_gates],
            "failure_criteria": list(self.failure_criteria),
            "replay_plan_id": self.replay_plan_id,
            "notes": list(self.notes),
            "decision": planned_decision(),
        }
        validate_experiment_record(record)
        return record


def planned_decision() -> dict[str, Any]:
    return {
        "status": "planned",
        "promoted": False,
        "summary": "Run intent recorded before training.",
        "decided_at": None,
        "evidence": [],
    }


def _require_non_empty_string(record: dict[str, Any], field_name: str) -> None:
    if not isinstance(record.get(field_name), str) or not record[field_name].strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_string_list(record: dict[str, Any], field_name: str) -> None:
    values = record.get(field_name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} must contain only non-empty strings")


def _validate_acceptance_gates(gates: Any) -> None:
    if not isinstance(gates, list) or not gates:
        raise ValueError("acceptance_gates must be a non-empty list")
    names: set[str] = set()
    for gate in gates:
        if not isinstance(gate, dict):
            raise ValueError("each acceptance gate must be an object")
        name = gate.get("name")
        rule = gate.get("rule")
        required = gate.get("required")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each acceptance gate needs a non-empty name")
        if name in names:
            raise ValueError(f"duplicate acceptance gate {name!r}")
        names.add(name)
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError(f"acceptance gate {name!r} needs a non-empty rule")
        if not isinstance(required, bool):
            raise ValueError(f"acceptance gate {name!r} needs a boolean required field")


def _validate_decision(decision: Any) -> None:
    if not isinstance(decision, dict):
        raise ValueError("decision must be an object")
    status = decision.get("status")
    if status not in DECISION_STATUSES:
        raise ValueError(f"decision status must be one of {sorted(DECISION_STATUSES)}")
    promoted = decision.get("promoted")
    if promoted is not (status == "promoted"):
        raise ValueError("decision.promoted must match promoted status")
    summary = decision.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("decision.summary must be a non-empty string")
    evidence = decision.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("decision.evidence must be a list")


def validate_experiment_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != EXPERIMENT_INTENT_SCHEMA_VERSION:
        raise ValueError("unsupported experiment intent schema_version")
    if record.get("kind") != "experiment_intent":
        raise ValueError("kind must be experiment_intent")
    for field_name in (
        "created_at",
        "version",
        "run_id",
        "component",
        "hypothesis",
        "training_recipe_id",
    ):
        _require_non_empty_string(record, field_name)
    for field_name in ("allowed_data_sources", "planned_artifacts", "failure_criteria"):
        _require_string_list(record, field_name)
    if record.get("replay_plan_id") is not None and not isinstance(
        record["replay_plan_id"], str
    ):
        raise ValueError("replay_plan_id must be a string or null")
    if not isinstance(record.get("notes"), list):
        raise ValueError("notes must be a list")
    _validate_acceptance_gates(record.get("acceptance_gates"))
    _validate_decision(record.get("decision"))


def write_experiment_intent(path: Path, record: dict[str, Any]) -> None:
    validate_experiment_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_experiment_intent(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    validate_experiment_record(record)
    return record


def record_experiment_decision(
    record: dict[str, Any],
    status: str,
    summary: str,
    evidence: list[dict[str, Any]] | None = None,
    decided_at: str | None = None,
) -> dict[str, Any]:
    updated = dict(record)
    updated["decision"] = {
        "status": status,
        "promoted": status == "promoted",
        "summary": summary,
        "decided_at": decided_at or utc_now_iso(),
        "evidence": list(evidence or []),
    }
    validate_experiment_record(updated)
    return updated
