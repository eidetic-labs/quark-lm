"""Combined epistemic report for closed-world evaluation.

Unifies the independent epistemic metrics into one report and a flat headline so
a run can be judged at a glance: did the model actually learn (NLL below the
random floor), does it abstain correctly on out-of-corpus queries, is it
calibrated, and how does it compare to the rule-based closed-world oracle.

The metrics consume already-scored evaluation data:
- ``evals``: per-eval-set summary dicts (carrying ``avg_target_nll``/``count``).
- ``scored_by_set``: per-eval-set lists of per-example scored records (carrying
  ``target``/``predicted_candidate``/``candidate_scores``/``candidate_match``).
"""

from __future__ import annotations

from typing import Any

from epistemic_abstention import abstention_metrics
from epistemic_calibration import expected_calibration_error
from epistemic_nll_vs_random import nll_vs_random
from epistemic_oracle_benchmark import oracle_benchmark


def epistemic_report(
    evals: dict[str, Any],
    scored_by_set: dict[str, list[dict[str, Any]]],
    vocab_size: int,
    responder: Any | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all epistemic metrics and a flat headline summary."""

    learning = nll_vs_random(evals, vocab_size)
    abstention = abstention_metrics(scored_by_set)
    calibration = expected_calibration_error(scored_by_set)
    pools = _candidate_pools_by_type(scored_by_set)
    chance_by_type = {answer_type: 1.0 / size for answer_type, size in pools.items()}
    mean_chance = (
        sum(chance_by_type.values()) / len(chance_by_type) if chance_by_type else None
    )
    # Menu-free correctness anchor: free-generation exact match, which a closed
    # candidate menu cannot inflate. Aggregated over all scored probes.
    total_count = sum(int(summary.get("count", 0)) for summary in evals.values())
    total_exact = sum(int(summary.get("exact", 0)) for summary in evals.values())
    generation_exact_rate = (total_exact / total_count) if total_count else None

    report: dict[str, Any] = {
        "nll_vs_random": learning,
        "abstention": abstention,
        "calibration": calibration,
        "candidate_pool": {"by_type": pools, "chance_by_type": chance_by_type},
        "provenance": provenance or {},
    }
    oracle = None
    if responder is not None:
        oracle = oracle_benchmark(scored_by_set, responder)
        report["oracle"] = oracle

    overall = learning.get("overall", {})
    report["headline"] = {
        "learned_all": overall.get("learned_all"),
        "learned_any": overall.get("learned_any"),
        "mean_nll_reduction": overall.get("mean_reduction"),
        "generation_exact_rate": generation_exact_rate,
        "abstention_f1": abstention.get("f1"),
        "abstention_precision": abstention.get("precision"),
        "abstention_recall": abstention.get("recall"),
        "calibration_ece": calibration.get("ece"),
        "oracle_exact_rate": (
            oracle.get("overall", {}).get("oracle_exact_rate")
            if oracle is not None
            else None
        ),
        "candidate_pool_by_type": pools,
        "mean_chance_accuracy": mean_chance,
    }
    return report


def _candidate_pools_by_type(scored_by_set: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Menu size per answer type (the de-contaminated per-type candidate pools).

    With per-type menus each question is ranked only against its type's answers +
    the abstain token, so chance = 1/size is reported per type rather than as one
    inflated global pool. Records without an answer_type bucket under "all".
    """

    pools: dict[str, int] = {}
    for records in scored_by_set.values():
        for record in records:
            scores = record.get("candidate_scores")
            if not scores:
                continue
            answer_type = record.get("answer_type") or "all"
            pools.setdefault(answer_type, len(scores))
    return pools
