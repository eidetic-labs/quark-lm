"""Supported-profile checks for minimal PyTorch transformer parity."""

from __future__ import annotations

from typing import Any


def minimal_forward_unsupported_reason(config: dict[str, Any]) -> str | None:
    unsupported_flags: list[str] = []
    for flag in unsupported_flags:
        if config.get(flag):
            return f"minimal PyTorch parity does not support {flag}"
    return None


# Profiles the Tier-2 batched (B,C,D) forward does NOT yet reproduce in parity.
# use_prompt_position_projection is position-indexed (a distinct W[position] per
# kept prompt position); a position-collapsing batched reduction would lose that
# indexing and miscompute. Until the per-position vectorization lands and is
# parity-verified, fail closed to the per-position Tier-1 path.
#
# use_absolute_rope (Phase 1): the batched _apply_rotary_batched keys RoPE by
# torch.arange (slot-keyed) and _attention never reads runtime['abs_positions'], so
# running it under the flag would silently rotate by slot while the scalar/Tier-1
# path rotates by absolute position -- a parity break on any padded window. Phase 1
# does NOT wire the batched absolute path; fail it closed to Tier-1 instead.
BATCHED_FORWARD_UNSUPPORTED_FLAGS: tuple[str, ...] = (
    "use_prompt_position_projection",
    "use_absolute_rope",
)


def batched_forward_unsupported_reason(config: dict[str, Any]) -> str | None:
    """Return a reason (non-None) when the batched forward must fall back to Tier-1.

    A non-None reason routes ``build_torch_training_logits`` to the per-position
    ``torch_minimal_logits`` instead of the batched path -- fail-closed rather than
    risk a miscompute on a not-yet-covered profile.
    """

    for flag in BATCHED_FORWARD_UNSUPPORTED_FLAGS:
        if config.get(flag):
            return f"batched PyTorch forward does not support {flag}"
    return None
