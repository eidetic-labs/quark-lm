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
# indexing and miscompute. The slot-keyed projection still cannot vectorize, so the
# thesis (projection-on) model must stay on the per-position Tier-1 path -- fail it
# closed.
#
# use_absolute_rope was LIFTED here: the batched _apply_rotary_batched now keys RoPE
# absolutely from runtime['abs_positions'] (threaded by the batched training/loss
# path, fail-closed RAISE if missing), parity-verified against Tier-1 over padded and
# odd-head_dim probes. The general-LM (projection-off) absolute-RoPE forward runs
# vectorized on Tier-2.
BATCHED_FORWARD_UNSUPPORTED_FLAGS: tuple[str, ...] = (
    "use_prompt_position_projection",
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
