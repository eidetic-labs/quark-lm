"""Compare per-eval-set target NLL against the random-chance floor ln(vocab).

This automates the "did the model actually learn?" check. A model that learned
has ``avg_target_nll`` well below ``ln(vocab_size)``; a degenerate or
mode-collapsed model sits at or above that floor.

Pure stdlib (``math`` only): no file or network I/O, no new dependencies,
fully deterministic.
"""

from __future__ import annotations

import math


def _random_floor(vocab_size: int) -> float:
    """Negative log-likelihood of uniform guessing over ``vocab_size`` tokens.

    ``ln(vocab_size)`` for a vocabulary larger than one token; ``0.0`` when the
    vocabulary is degenerate (a single token, or fewer), where there is no
    uncertainty to reduce.
    """

    if vocab_size > 1:
        return math.log(vocab_size)
    return 0.0


def nll_vs_random(evals: dict, vocab_size: int) -> dict:
    """Score each eval set's target NLL against the random-chance floor.

    Args:
        evals: Mapping of eval-set name -> metric dict. Each metric dict may
            contain ``"avg_target_nll"`` (float) and ``"count"`` (int). Sets
            without ``"avg_target_nll"`` are skipped gracefully.
        vocab_size: Vocabulary size used to derive the random floor ``ln(vocab)``.

    Returns:
        A dict with the random floor, the resolved vocab size, a ``per_set``
        breakdown of scored sets, and an ``overall`` summary. The ``overall``
        figures are count-weighted across the scored sets.
    """

    random_floor = _random_floor(vocab_size)

    per_set: dict = {}
    weighted_nll_numerator = 0.0
    weighted_reduction_numerator = 0.0
    total_weight = 0.0
    learned_flags: list = []

    for name, metrics in evals.items():
        if not isinstance(metrics, dict):
            continue
        if "avg_target_nll" not in metrics:
            continue
        avg_target_nll = float(metrics["avg_target_nll"])

        if random_floor > 0.0:
            reduction = (random_floor - avg_target_nll) / random_floor
        else:
            reduction = 0.0
        learned = avg_target_nll < random_floor

        per_set[name] = {
            "avg_target_nll": avg_target_nll,
            "random_floor": random_floor,
            "reduction": reduction,
            "learned": learned,
        }

        # Count-weighted accumulation; default to weight 1.0 when count absent
        # or non-positive so a scored set is never silently dropped from the
        # overall figures.
        count = metrics.get("count", 1)
        try:
            weight = float(count)
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0.0:
            weight = 1.0

        weighted_nll_numerator += avg_target_nll * weight
        weighted_reduction_numerator += reduction * weight
        total_weight += weight
        learned_flags.append(learned)

    sets_scored = len(per_set)
    if total_weight > 0.0:
        weighted_avg_target_nll = weighted_nll_numerator / total_weight
        mean_reduction = weighted_reduction_numerator / total_weight
    else:
        weighted_avg_target_nll = 0.0
        mean_reduction = 0.0

    overall = {
        "weighted_avg_target_nll": weighted_avg_target_nll,
        "mean_reduction": mean_reduction,
        "learned_all": sets_scored > 0 and all(learned_flags),
        "learned_any": any(learned_flags),
        "sets_scored": sets_scored,
    }

    return {
        "random_floor": random_floor,
        "vocab_size": vocab_size,
        "per_set": per_set,
        "overall": overall,
    }
