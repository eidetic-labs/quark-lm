"""Trainable-parameter manifests for transformer training parity."""

from __future__ import annotations

from typing import Any


TRAINING_PARAMETER_MANIFEST_SCHEMA_VERSION = 1
TRAINING_PARAMETER_ORDER = "scalar_optimizer_all_transformer_parameters_v1"


def build_training_parameter_manifest(
    *,
    weights: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Describe scalar optimizer parameter order for backend parity."""

    builder = _ManifestBuilder()
    _add_base_parameters(builder, weights)
    _add_optional_global_parameters(builder, weights, model_config)
    _add_layer_norm_parameters(builder, weights, model_config)
    _add_extra_layers(builder, weights, model_config)
    return builder.to_manifest(model_config)


def validate_training_parameter_manifest(
    manifest: dict[str, Any],
    *,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    """Validate a training parameter manifest before treating it as evidence."""

    if manifest.get("schema_version") != TRAINING_PARAMETER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported training parameter manifest schema_version")
    if manifest.get("parameter_order") != TRAINING_PARAMETER_ORDER:
        raise ValueError("unsupported training parameter order")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("parameter manifest entries must be a list")
    cursor = 0
    for entry in entries:
        cursor = _validate_entry(entry, cursor)
    if manifest.get("parameter_count") != cursor:
        raise ValueError("parameter manifest count does not match entries")
    if manifest.get("tensor_count") != len(entries):
        raise ValueError("parameter manifest tensor_count does not match entries")
    if optimizer_state is not None and optimizer_state.get("param_count") != cursor:
        raise ValueError("parameter manifest count does not match optimizer state")


class _ManifestBuilder:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.cursor = 0

    def add(self, name: str, value: Any) -> None:
        shape = _shape(value)
        count = _element_count(shape)
        self.entries.append(
            {
                "name": name,
                "shape": shape,
                "count": count,
                "index_start": self.cursor,
                "index_end": self.cursor + count,
            }
        )
        self.cursor += count

    def to_manifest(self, model_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_PARAMETER_MANIFEST_SCHEMA_VERSION,
            "parameter_order": TRAINING_PARAMETER_ORDER,
            "parameter_count": self.cursor,
            "tensor_count": len(self.entries),
            "tie_output_embeddings": bool(
                model_config.get("tie_output_embeddings", False)
            ),
            "entries": self.entries,
        }


def _add_base_parameters(
    builder: _ManifestBuilder,
    weights: dict[str, Any],
) -> None:
    for name in (
        "token_embeddings",
        "position_embeddings",
        "wq",
        "bq",
        "wk",
        "bk",
        "wv",
        "bv",
        "wo",
        "bo",
        "w1",
        "b1",
        "w2",
        "b2",
        "bout",
    ):
        builder.add(name, weights[name])


def _add_optional_global_parameters(
    builder: _ManifestBuilder,
    weights: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    if model_config.get("use_gated_mlp", False):
        for name in ("w_gate", "b_gate"):
            builder.add(name, weights[name])
    if not model_config.get("tie_output_embeddings", False):
        builder.add("wout", weights["wout"])
    if model_config.get("use_context_projection", False):
        for name in ("context_projection_w", "context_projection_b"):
            builder.add(name, weights[name])
    if model_config.get("use_prompt_prefix_projection", False):
        for name in ("prompt_prefix_projection_w", "prompt_prefix_projection_b"):
            builder.add(name, weights[name])
    if model_config.get("use_prompt_position_projection", False):
        for name in ("prompt_position_projection_w", "prompt_position_projection_b"):
            builder.add(name, weights[name])
    if model_config.get("use_prompt_attention_summary", False):
        for name in ("prompt_summary_query", "prompt_summary_w", "prompt_summary_b"):
            builder.add(name, weights[name])


def _add_layer_norm_parameters(
    builder: _ManifestBuilder,
    weights: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    if model_config.get("use_layer_norm", False) or model_config.get(
        "use_pre_layer_norm",
        False,
    ):
        for name in ("ln1_gain", "ln1_bias", "ln2_gain", "ln2_bias"):
            builder.add(name, weights[name])
    if model_config.get("use_pre_layer_norm", False):
        for name in ("final_ln_gain", "final_ln_bias"):
            builder.add(name, weights[name])


def _add_extra_layers(
    builder: _ManifestBuilder,
    weights: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    for layer_index, layer in enumerate(weights.get("extra_layers", [])):
        _add_extra_layer(builder, layer, layer_index, model_config)


def _add_extra_layer(
    builder: _ManifestBuilder,
    layer: dict[str, Any],
    layer_index: int,
    model_config: dict[str, Any],
) -> None:
    prefix = f"extra_layers.{layer_index}"
    for name in (
        "wq",
        "bq",
        "wk",
        "bk",
        "wv",
        "bv",
        "wo",
        "bo",
        "w1",
        "b1",
        "w2",
        "b2",
    ):
        builder.add(f"{prefix}.{name}", layer[name])
    if model_config.get("use_gated_mlp", False):
        for name in ("w_gate", "b_gate"):
            builder.add(f"{prefix}.{name}", layer[name])
    if model_config.get("use_layer_norm", False) or model_config.get(
        "use_pre_layer_norm",
        False,
    ):
        for name in ("ln1_gain", "ln1_bias", "ln2_gain", "ln2_bias"):
            builder.add(f"{prefix}.{name}", layer[name])


def _shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    first_shape = _shape(value[0])
    for item in value[1:]:
        if _shape(item) != first_shape:
            raise ValueError("parameter tensors must be rectangular")
    return [len(value), *first_shape]


def _element_count(shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _validate_entry(entry: Any, expected_start: int) -> int:
    if not isinstance(entry, dict):
        raise ValueError("parameter manifest entry must be a dict")
    if not entry.get("name"):
        raise ValueError("parameter manifest entry name is required")
    count = entry.get("count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("parameter manifest entry count must be positive")
    shape = entry.get("shape")
    if not _valid_shape(shape):
        raise ValueError("parameter manifest entry shape must be a list of integers")
    if entry.get("index_start") != expected_start:
        raise ValueError("parameter manifest entry index_start is not contiguous")
    expected_end = expected_start + count
    if entry.get("index_end") != expected_end:
        raise ValueError("parameter manifest entry index_end is incorrect")
    if _element_count(shape) != count:
        raise ValueError("parameter manifest entry shape does not match count")
    return expected_end


def _valid_shape(shape: Any) -> bool:
    return isinstance(shape, list) and all(
        isinstance(dimension, int) and dimension >= 0
        for dimension in shape
    )
