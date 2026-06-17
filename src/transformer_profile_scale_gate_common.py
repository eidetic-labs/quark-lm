"""Shared helpers for profile-scale experiment gates."""

from __future__ import annotations

from typing import Any


def required_gate(name: str, rule: str) -> dict[str, Any]:
    return {"name": name, "rule": rule, "required": True}
