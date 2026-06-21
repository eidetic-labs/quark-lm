"""Validated-tolerance parity contract for the torch performance backend (Phase 1).

Decisions D1/D2 of the training-optimization plan: the torch backend is *validated*
against the bit-exact scalar reference, not held to per-op float64 identity (which
is what made it slow and blocks a float32/GPU path). The contract has two parts:

  * a numeric tolerance band on a per-quantity value, parameterized by dtype, and
  * a ZERO-TOLERANCE rank-invariance check.

Rank-invariance is the load-bearing assertion. Candidate accuracy depends ONLY on
the argmin-NLL ordering, so the torch backend must rank candidates identically to
scalar -- with a pinned, stable tie-break (ties resolved by candidate index) so the
decision is deterministic even when float32 makes near-ties more likely. A loose
epsilon on raw NLLs would silently launder a ranking divergence, which is exactly
the accuracy claim the closed-world thesis sells.
"""

from __future__ import annotations

from collections.abc import Sequence

# Per-dtype numeric tolerance band for a single teacher-forced sequence NLL.
# float64 mirrors the established scalar<->torch parity tolerance; the float32 and
# bfloat16 bands are derived to bound on-device drift without admitting rank flips.
TOLERANCES: dict[str, dict[str, float]] = {
    "float64": {"abs": 1e-6, "rel": 1e-6},
    "float32": {"abs": 3e-3, "rel": 3e-3},
    "bfloat16": {"abs": 5e-2, "rel": 5e-2},
}


def candidate_ranking(nlls: Sequence[float]) -> list[int]:
    """Stable argmin-first candidate order; ties resolved by lower index (pinned)."""

    return sorted(range(len(nlls)), key=lambda index: (nlls[index], index))


def numeric_parity_violations(
    scalar: Sequence[float], torch_values: Sequence[float], *, dtype: str
) -> list[tuple[int, float, float]]:
    """Indices where torch deviates from scalar beyond the dtype tolerance band."""

    if dtype not in TOLERANCES:
        raise ValueError(f"no parity tolerance defined for dtype {dtype!r}")
    if len(scalar) != len(torch_values):
        raise ValueError("scalar and torch value counts differ")
    tol = TOLERANCES[dtype]
    violations: list[tuple[int, float, float]] = []
    for index, (s, t) in enumerate(zip(scalar, torch_values)):
        if abs(s - t) > tol["abs"] + tol["rel"] * abs(s):
            violations.append((index, s, t))
    return violations


def assert_numeric_parity(
    scalar: Sequence[float], torch_values: Sequence[float], *, dtype: str
) -> None:
    violations = numeric_parity_violations(scalar, torch_values, dtype=dtype)
    if violations:
        raise AssertionError(
            f"numeric parity outside {dtype} tolerance at {violations[:5]}"
        )


def assert_rank_invariant(
    scalar_nlls: Sequence[float], torch_nlls: Sequence[float]
) -> None:
    """Zero-tolerance: torch must rank candidates exactly as scalar does."""

    scalar_order = candidate_ranking(scalar_nlls)
    torch_order = candidate_ranking(torch_nlls)
    if scalar_order != torch_order:
        raise AssertionError(
            f"rank-invariance violated: scalar order {scalar_order} "
            f"!= torch order {torch_order}"
        )


def validate_candidate_parity(
    scalar_nlls_per_probe: Sequence[Sequence[float]],
    torch_nlls_per_probe: Sequence[Sequence[float]],
    *,
    dtype: str,
) -> None:
    """Apply both contract checks across a set of candidate-ranking probes."""

    if len(scalar_nlls_per_probe) != len(torch_nlls_per_probe):
        raise ValueError("probe counts differ")
    for scalar_nlls, torch_nlls in zip(scalar_nlls_per_probe, torch_nlls_per_probe):
        assert_numeric_parity(scalar_nlls, torch_nlls, dtype=dtype)
        assert_rank_invariant(scalar_nlls, torch_nlls)
