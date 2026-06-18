"""Supported-profile checks for minimal PyTorch transformer parity."""

from __future__ import annotations

from typing import Any


def minimal_forward_unsupported_reason(config: dict[str, Any]) -> str | None:
    unsupported_flags: list[str] = []
    for flag in unsupported_flags:
        if config.get(flag):
            return f"minimal PyTorch parity does not support {flag}"
    return None
