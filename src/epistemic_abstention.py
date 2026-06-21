"""Epistemic abstention metric.

The closed-world LM's core thesis is reliable abstention: when a query falls
outside the admitted corpus, the model should answer with the literal string
``" unknown."`` rather than confabulate. This module scores how well the model
distinguishes should-abstain (out-of-corpus) queries from in-corpus ones.

Abstention is treated as the *positive* class:

    tp  should abstain & model abstained   (correctly said " unknown.")
    fp  in-corpus      & model abstained   (wrongly refused to answer)
    fn  should abstain & model answered    (confabulated instead of abstaining)
    tn  in-corpus      & model answered     (correctly attempted an answer)

The module is pure: stdlib only, deterministic, no I/O.
"""

from __future__ import annotations

from typing import Any

ABSTAIN_TARGET = " unknown."
# When ``predicted_candidate`` is None we fall back to the greedy completion,
# which carries no leading space, so the abstention token is stored without one.
ABSTAIN_COMPLETION = "unknown."

# Ledgered abstention calibration. The decision is pure argmin over the per-type
# candidate menu -- the model "abstains" when ``" unknown."`` is the lowest-NLL
# choice (or the greedy completion is ``"unknown."``) -- i.e. margin 0.0: no
# confidence band beyond winning the ranking is required. This constant is the
# calibration knob: raising it would require the abstain option to beat the best
# concrete answer by a margin (trading recall for precision). Changing it is a
# deliberate, reviewed decision -- pinned by tests/test_abstention_ledger.py and
# emitted into eval provenance so any recalibration is auditable.
ABSTENTION_MARGIN = 0.0
ABSTENTION_DECISION = "argmin_nll_over_per_type_menu_or_greedy_completion"


def abstention_ledger() -> dict[str, Any]:
    """The pinned abstention decision contract -- the closed-world thesis's
    load-bearing definition -- emitted into eval provenance for auditability."""

    return {
        "target": ABSTAIN_TARGET,
        "completion": ABSTAIN_COMPLETION,
        "decision": ABSTENTION_DECISION,
        "margin": ABSTENTION_MARGIN,
    }


def _should_abstain(record: dict) -> bool:
    """A record is out-of-corpus when its gold target is the abstain string."""
    return record.get("target") == ABSTAIN_TARGET


def _model_abstained(record: dict) -> bool:
    """Did the model abstain?

    Prefer the explicit ``predicted_candidate``. When it is ``None`` (no
    candidate set was supplied), fall back to the greedy ``completion``. Any
    other value counts as not-abstained.
    """
    predicted = record.get("predicted_candidate")
    if predicted is not None:
        return predicted == ABSTAIN_TARGET
    completion = record.get("completion", "")
    if not isinstance(completion, str):
        return False
    return completion.strip() == ABSTAIN_COMPLETION


def _rates(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Precision / recall / f1 for the abstain-positive confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1: float | None = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _empty_confusion() -> dict[str, int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "total": 0,
        "should_abstain": 0,
        "model_abstained": 0,
    }


def _accumulate(counts: dict[str, int], record: dict) -> None:
    """Fold a single record into a confusion-count accumulator in place."""
    should = _should_abstain(record)
    abstained = _model_abstained(record)
    counts["total"] += 1
    if should:
        counts["should_abstain"] += 1
    if abstained:
        counts["model_abstained"] += 1
    if should and abstained:
        counts["tp"] += 1
    elif not should and abstained:
        counts["fp"] += 1
    elif should and not abstained:
        counts["fn"] += 1
    else:  # not should and not abstained
        counts["tn"] += 1


def abstention_metrics(scored_by_set: dict) -> dict[str, Any]:
    """Measure abstention quality across scored eval records.

    Args:
        scored_by_set: mapping of eval-set name -> list of scored record dicts.
            Each record carries at least ``target`` (gold answer; the string
            ``" unknown."`` marks an out-of-corpus query) and either
            ``predicted_candidate`` or ``completion``.

    Returns:
        A dict with global ``counts`` (tp/fp/fn/tn/total/should_abstain/
        model_abstained), global ``precision``/``recall``/``f1`` (any of which
        may be ``None`` when undefined), and ``per_set`` mapping each set name
        to its own confusion counts plus rates.

    Empty or missing input is handled without raising.
    """
    overall = _empty_confusion()
    per_set: dict[str, dict[str, Any]] = {}

    for name, records in (scored_by_set or {}).items():
        set_counts = _empty_confusion()
        for record in records or []:
            _accumulate(set_counts, record)
            _accumulate(overall, record)
        set_entry: dict[str, Any] = dict(set_counts)
        set_entry.update(
            _rates(set_counts["tp"], set_counts["fp"], set_counts["fn"])
        )
        per_set[name] = set_entry

    result: dict[str, Any] = {"counts": overall}
    result.update(_rates(overall["tp"], overall["fp"], overall["fn"]))
    result["per_set"] = per_set
    return result
