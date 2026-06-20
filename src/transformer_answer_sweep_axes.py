"""Controlled axis expansion for answer-training sweeps."""

from __future__ import annotations

import copy
from argparse import Namespace
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

from transformer_objectives import DIRECT_ANSWER_OBJECTIVE_MODES
from transformer_profiles import DEFAULT_PROFILE, MODERN_SMALL_PROFILE
from transformer_routing_repair_bundle import (
    EXPERIMENT_BUNDLES,
    routing_repair_bundle_mode,
)


@dataclass(frozen=True)
class SweepTrial:
    trial_id: str
    config: dict[str, Any]
    args: Namespace


_STRING_CHOICES = {
    "tokenizer": {"char", "closed-world-subword"},
    "transformer_profile": {DEFAULT_PROFILE, MODERN_SMALL_PROFILE},
    "optimizer": {"sgd", "adamw"},
    "direct_answer_mode": set(DIRECT_ANSWER_OBJECTIVE_MODES),
    "experiment_bundle": set(EXPERIMENT_BUNDLES),
}

_VALUE_PARSERS: dict[str, Callable[[str], Any]] = {
    "tokenizer": str,
    "transformer_profile": str,
    "context_size": int,
    "embedding_dim": int,
    "attention_heads": int,
    "num_layers": int,
    "feedforward_dim": int,
    "optimizer": str,
    "steps": int,
    "direct_answer_steps": int,
    "direct_answer_mode": str,
    "experiment_bundle": str,
    "learning_rate": float,
    "direct_answer_learning_rate": float,
}


def parse_sweep_axes(specs: list[str] | None) -> dict[str, list[Any]]:
    axes: dict[str, list[Any]] = {}
    for spec in specs or []:
        name, values = _split_axis_spec(spec)
        axes[name] = [_parse_axis_value(name, value) for value in values]
    return axes


def build_sweep_trials(args: Namespace, axes: dict[str, list[Any]]) -> list[SweepTrial]:
    if not axes:
        return [_trial_from_config(args, {}, 1)]
    names = list(axes)
    configs = [
        dict(zip(names, values, strict=True))
        for values in product(*(axes[name] for name in names))
    ]
    return [
        _trial_from_config(args, config, index)
        for index, config in enumerate(configs, start=1)
    ]


def sweep_axes_summary(axes: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "axis_count": len(axes),
        "axes": axes,
        "trial_count": _trial_count(axes),
    }


def _split_axis_spec(spec: str) -> tuple[str, list[str]]:
    if "=" not in spec:
        raise ValueError("sweep axis must be formatted as name=value[,value]")
    name, raw_values = spec.split("=", 1)
    name = name.strip().replace("-", "_")
    if name not in _VALUE_PARSERS:
        supported = ", ".join(sorted(_VALUE_PARSERS))
        raise ValueError(f"unsupported sweep axis {name!r}; supported: {supported}")
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError(f"sweep axis {name!r} must include at least one value")
    return name, values


def _parse_axis_value(name: str, value: str) -> Any:
    parser = _VALUE_PARSERS[name]
    try:
        parsed = parser(value)
    except ValueError as exc:
        raise ValueError(f"invalid value {value!r} for sweep axis {name!r}") from exc
    choices = _STRING_CHOICES.get(name)
    if choices is not None and parsed not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"invalid value {parsed!r} for {name!r}; allowed: {allowed}")
    return parsed


def _trial_from_config(args: Namespace, config: dict[str, Any], index: int) -> SweepTrial:
    trial_args = copy.copy(args)
    trial_config = dict(config)
    for name, value in config.items():
        setattr(trial_args, name, value)
    derived_mode = _apply_bundle_default_mode(trial_args, config)
    if derived_mode is not None:
        trial_config["direct_answer_mode"] = derived_mode
    trial_args.command = "answer-train"
    trial_args.run = args.run / _trial_slug(index, trial_config)
    trial_args.tokenizer_manifest = None
    trial_args.tokenizer_report = None
    return SweepTrial(
        trial_id=f"trial-{index:02d}",
        config=trial_config,
        args=trial_args,
    )


def _apply_bundle_default_mode(args: Namespace, config: dict[str, Any]) -> str | None:
    if "experiment_bundle" not in config or "direct_answer_mode" in config:
        return None
    if getattr(args, "direct_answer_mode", None) != "first-error":
        return None
    bundle_mode = routing_repair_bundle_mode(config["experiment_bundle"])
    if bundle_mode is not None:
        args.direct_answer_mode = bundle_mode
        return bundle_mode
    return None


def _trial_slug(index: int, config: dict[str, Any]) -> str:
    parts = [f"trial-{index:02d}"]
    for name, value in sorted(config.items()):
        parts.append(f"{name}-{_slug_value(value)}")
    return "__".join(parts)


def _slug_value(value: Any) -> str:
    return str(value).replace("_", "-").replace(".", "p")


def _trial_count(axes: dict[str, list[Any]]) -> int:
    total = 1
    for values in axes.values():
        total *= len(values)
    return total
