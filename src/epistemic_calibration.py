"""Expected Calibration Error for closed-world eval records.

A closed-world model's value is knowing *when* it knows. This module measures
whether the model's confidence matches its observed accuracy, summarised as the
Expected Calibration Error (ECE).

Confidence per record is a per-token geometric-mean probability in (0, 1],
recovered from a negative log-likelihood as ``exp(-min_nll)``. Correctness is
whether the predicted candidate matched the gold target.

Pure, deterministic, stdlib ``math`` only. No I/O, no new dependencies.
"""

from __future__ import annotations

import math


def _record_confidence(record: dict):
    """Return confidence in (0, 1] for a record, or ``None`` if unavailable.

    Confidence = ``exp(-min_nll)`` where ``min_nll`` is the smallest
    ``target_nll`` over ``candidate_scores`` (the predicted candidate is the
    one with the lowest NLL). If ``candidate_scores`` is absent or empty, fall
    back to the record-level ``target_nll``. If neither is present, return
    ``None`` so the caller can skip the record.
    """
    candidate_scores = record.get("candidate_scores")
    if candidate_scores:
        nlls = [
            candidate["target_nll"]
            for candidate in candidate_scores
            if "target_nll" in candidate
        ]
        if nlls:
            return math.exp(-min(nlls))
    if record.get("target_nll") is not None:
        return math.exp(-record["target_nll"])
    return None


def _record_correct(record: dict) -> int:
    """Return 1 if the record's prediction is correct, else 0.

    Prefer ``candidate_match`` when it is not ``None``; otherwise fall back to
    ``exact_match`` (defaulting to ``False``).
    """
    match = record.get("candidate_match")
    if match is None:
        match = record.get("exact_match", False)
    return 1 if match else 0


def _pairs(scored_by_set: dict):
    """Flatten all records to ``(confidence, correct)`` pairs.

    Records lacking both ``candidate_scores`` and ``target_nll`` (i.e. those
    with no recoverable confidence) are skipped.
    """
    pairs = []
    for records in scored_by_set.values():
        for record in records:
            confidence = _record_confidence(record)
            if confidence is None:
                continue
            # Clamp into [0, 1] to guard against tiny floating-point overshoot
            # (e.g. a marginally negative nll producing exp(-nll) > 1).
            confidence = min(1.0, max(0.0, confidence))
            pairs.append((confidence, _record_correct(record)))
    return pairs


def expected_calibration_error(scored_by_set: dict, n_bins: int = 10) -> dict:
    """Compute the Expected Calibration Error over scored eval records.

    Args:
        scored_by_set: mapping of set name -> list of scored eval record dicts.
        n_bins: number of equal-width confidence bins partitioning [0, 1].

    Returns:
        A dict with keys:
          ``ece``: float in [0, 1], the count-weighted mean gap between bin
              average confidence and bin accuracy. ``0.0`` for empty input.
          ``n``: int, number of records contributing a confidence/correct pair.
          ``n_bins``: int, the bin count used.
          ``bins``: list of non-empty bins, each
              ``{"lo", "hi", "count", "avg_confidence", "accuracy"}``,
              ordered low to high.
    """
    pairs = _pairs(scored_by_set)
    n = len(pairs)
    if n == 0:
        return {"ece": 0.0, "n": 0, "n_bins": n_bins, "bins": []}

    width = 1.0 / n_bins
    # Accumulate confidence sum and correct count per bin index.
    conf_sums = [0.0] * n_bins
    correct_counts = [0] * n_bins
    counts = [0] * n_bins

    for confidence, correct in pairs:
        # Assign to a bin; confidence == 1.0 lands in the last bin.
        index = int(confidence / width)
        if index >= n_bins:
            index = n_bins - 1
        conf_sums[index] += confidence
        correct_counts[index] += correct
        counts[index] += 1

    ece = 0.0
    bins = []
    for index in range(n_bins):
        count = counts[index]
        if count == 0:
            continue
        avg_confidence = conf_sums[index] / count
        accuracy = correct_counts[index] / count
        ece += (count / n) * abs(avg_confidence - accuracy)
        bins.append(
            {
                "lo": index * width,
                "hi": (index + 1) * width,
                "count": count,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
            }
        )

    return {"ece": ece, "n": n, "n_bins": n_bins, "bins": bins}
