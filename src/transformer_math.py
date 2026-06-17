"""Compatibility exports for transformer math helpers."""

from __future__ import annotations

from transformer_linear_ops import (
    dot_floats,
    dot_scalars,
    linear_floats,
    linear_scalars,
)
from transformer_nll import average_nll
from transformer_norms import (
    layer_norm_floats,
    layer_norm_scalars,
    rms_norm_floats,
    rms_norm_scalars,
)
from transformer_probabilities import (
    cross_entropy_scalars,
    softmax_floats,
    softmax_scalars,
)
from transformer_sampling import generation_distribution, sample_from_probs
from transformer_scalar_values import (
    exclude_scalars,
    flatten_scalars,
    matrix_to_floats,
    matrix_to_scalars,
    vector_to_floats,
    vector_to_scalars,
)


__all__ = [
    "average_nll",
    "cross_entropy_scalars",
    "dot_floats",
    "dot_scalars",
    "exclude_scalars",
    "flatten_scalars",
    "generation_distribution",
    "layer_norm_floats",
    "layer_norm_scalars",
    "linear_floats",
    "linear_scalars",
    "matrix_to_floats",
    "matrix_to_scalars",
    "rms_norm_floats",
    "rms_norm_scalars",
    "sample_from_probs",
    "softmax_floats",
    "softmax_scalars",
    "vector_to_floats",
    "vector_to_scalars",
]
