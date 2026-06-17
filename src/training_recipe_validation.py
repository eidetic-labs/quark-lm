"""Validation helpers shared by training recipe artifacts."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = 1


def require_non_empty_string(record: dict[str, Any], field_name: str) -> None:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def require_string_list(record: dict[str, Any], field_name: str) -> None:
    values = record.get(field_name)
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in values
    ):
        raise ValueError(f"{field_name} must contain only strings")


def validate_named_rules(values: Any, field_name: str) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must contain dicts")
        for key in ("name", "rule"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"{field_name} entries need {key}")
        if not isinstance(value.get("required"), bool):
            raise ValueError(f"{field_name} entries need required bool")


def validate_checks(values: Any, field_name: str) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list")
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must contain dicts")
        for key in ("name", "status", "rule"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"{field_name} entries need {key}")
        if not isinstance(value.get("passed"), bool):
            raise ValueError(f"{field_name} entries need passed bool")
        if value["status"] != ("passed" if value["passed"] else "failed"):
            raise ValueError(f"{field_name} status must match passed")
        if not isinstance(value.get("details"), dict):
            raise ValueError(f"{field_name} entries need details dict")
