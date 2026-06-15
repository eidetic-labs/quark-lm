"""Transformer checkpoint loading and validation surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .transformer_model import (
    TRANSFORMER_ARCHITECTURE,
    TRANSFORMER_CHECKPOINT_FORMAT,
)


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    if payload.get("architecture") != TRANSFORMER_ARCHITECTURE:
        raise ValueError(f"checkpoint architecture must be {TRANSFORMER_ARCHITECTURE}")
    if payload.get("checkpoint_format") != TRANSFORMER_CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint_format must be {TRANSFORMER_CHECKPOINT_FORMAT}")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("checkpoint config must be a dict")
    if not isinstance(payload.get("weights"), dict):
        raise ValueError("checkpoint weights must be a dict")
    if "tokenizer" in payload and not isinstance(payload["tokenizer"], dict):
        raise ValueError("checkpoint tokenizer must be a dict when present")
    if "metadata" in payload and not isinstance(payload["metadata"], dict):
        raise ValueError("checkpoint metadata must be a dict when present")


def checkpoint_summary(payload: dict[str, Any]) -> dict[str, Any]:
    validate_checkpoint_payload(payload)
    config = payload["config"]
    weights = payload["weights"]
    return {
        "architecture": payload["architecture"],
        "checkpoint_format": payload["checkpoint_format"],
        "has_tokenizer": "tokenizer" in payload,
        "has_metadata": "metadata" in payload,
        "vocab_size": config.get("vocab_size"),
        "context_size": config.get("context_size"),
        "weight_groups": sorted(weights),
    }


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    validate_checkpoint_payload(payload)
    return payload
