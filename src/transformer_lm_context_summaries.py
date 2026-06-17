"""Combined final-context summary helpers for TinyTransformerLM."""

from __future__ import annotations

from transformer_lm_context_summary_floats import TransformerFloatContextSummaryMixin
from transformer_lm_context_summary_scalars import TransformerScalarContextSummaryMixin


class TransformerContextSummaryMixin(
    TransformerScalarContextSummaryMixin,
    TransformerFloatContextSummaryMixin,
):
    """Provide scalar and float final-context summary helpers."""
