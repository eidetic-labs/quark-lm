"""Parameter-signature helpers for training parity evidence."""

from __future__ import annotations

from typing import Any


def build_weight_tree_signature(weights: dict[str, Any]) -> dict[str, float | int]:
    """Summarize every numeric scalar in a model weight tree."""

    return _signature(list(_numbers(weights)))


def build_manifest_parameter_signature(
    *,
    weights: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, float | int]:
    """Summarize trainable parameters in manifest optimizer order."""

    values = [
        number
        for entry in manifest["entries"]
        for number in _numbers(_resolve_weight(weights, entry["name"]))
    ]
    return _signature(values)


def _resolve_weight(weights: dict[str, Any], name: str) -> Any:
    current: Any = weights
    for segment in name.split("."):
        current = current[int(segment)] if isinstance(current, list) else current[segment]
    return current


def _signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


def _numbers(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, int | float):
        yield float(value)
