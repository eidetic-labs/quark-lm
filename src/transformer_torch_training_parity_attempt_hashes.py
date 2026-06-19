"""Payload hashes for PyTorch training parity attempt artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM = "sha256-json-v1"
HASHED_TORCH_TRAINING_ATTEMPT_ARTIFACTS = ("fixture", "candidate", "report")


def build_torch_training_parity_attempt_hashes(
    artifacts: dict[str, Any],
) -> dict[str, str]:
    """Return canonical payload hashes for written attempt sibling artifacts."""

    return {
        name: _payload_hash(_required_payload(artifacts, name))
        for name in HASHED_TORCH_TRAINING_ATTEMPT_ARTIFACTS
    }


def build_torch_runtime_report_hash(runtime_report: dict[str, Any]) -> str:
    """Return the canonical payload hash for embedded runtime preflight evidence."""

    return build_torch_training_attempt_payload_hash(
        runtime_report,
        payload_name="runtime_report",
    )


def build_torch_training_attempt_payload_hash(
    payload: dict[str, Any],
    *,
    payload_name: str = "payload",
) -> str:
    """Return the canonical hash for a JSON-safe attempt evidence payload."""

    if not isinstance(payload, dict):
        raise ValueError(f"{payload_name} must be a dict")
    return _payload_hash(payload)


def _required_payload(artifacts: dict[str, Any], name: str) -> dict[str, Any]:
    value = artifacts.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"artifacts.{name} must be a dict")
    return value


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
