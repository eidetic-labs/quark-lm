"""Fail-closed contract for the unbounded RoPE + append-valid KV-cache architecture.

The architecture (absolute-keyed RoPE -> regime-aware append-valid cache ->
sliding-window attention -> optional all-positions causal-LM) lands in phases. Each
positional/cache flag is FAIL-CLOSED: enabling it raises until its phase is built and
parity-verified, so a half-wired flag can never silently no-op and still look green --
the exact regression class (a zero-update run reading as "passing") the recovery work
exists to prevent. As each phase lands, its flag is removed from ``_UNIMPLEMENTED``.

A SEPARATE, PERMANENT guard rejects the adversarially-refuted naive design: an
evicting K/V-only cache cannot reconstruct the ``use_prompt_position_projection``
readout, which reads the full right-anchored window's residual rows through a
per-window-slot table -- not K/V, and K = linear(x) is non-invertible. So
{evicting window + position-projection} is mathematically unconstructible no matter
how many phases have landed; pin the window to the full sequence length or disable the
projection. This guard outlives the per-phase entries.
"""

from __future__ import annotations

from typing import Any

# Flags whose forward/cache path is not yet implemented. Removed per phase as each
# lands with scalar+torch parity. Until then, enabling the flag fails closed.
_UNIMPLEMENTED: dict[str, str] = {
    "use_absolute_rope": "absolute-keyed RoPE is not yet implemented (Phase 1)",
    "kv_cache_stores_summary_state": (
        "the regime-aware append-valid KV cache is not yet implemented (Phase 3)"
    ),
    "use_all_positions_causal": (
        "all-positions causal-LM training is not yet implemented (Phase 5)"
    ),
}


def kv_cache_contract_violation(config: dict[str, Any]) -> str | None:
    """Return a reason (non-None) when a config requests an unbuilt or impossible path.

    The default config -- every architecture flag off -- returns ``None`` and is
    byte-for-byte the existing behavior. A non-None return is a reason string the
    caller raises on.
    """

    for flag, reason in _UNIMPLEMENTED.items():
        if config.get(flag):
            return reason
    if config.get("sliding_window_size") is not None:
        return "sliding-window attention is not yet implemented (Phase 4)"
    if _evicting_window_with_projection(config):
        return (
            "an evicting window is incompatible with use_prompt_position_projection: "
            "the projection reads the full right-anchored window's residual rows "
            "(not K/V), so an evicting cache cannot reconstruct them -- pin the "
            "window to the full sequence length or disable the projection"
        )
    return None


def _evicting_window_with_projection(config: dict[str, Any]) -> bool:
    return bool(
        config.get("sliding_window_size") is not None
        and config.get("use_prompt_position_projection")
    )
