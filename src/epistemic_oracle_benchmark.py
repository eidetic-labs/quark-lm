"""Benchmark eval records against the rule-based closed-world oracle.

The :class:`~corpus_responder.CorpusResponder` provides the gold closed-world
answer for any prompt: a real answer for in-corpus queries and ``" unknown."``
for out-of-corpus queries. Comparing each eval record's ``target`` to the
oracle's answer measures the *achievable closed-world ceiling* -- the best a
perfectly faithful model could do given the admitted corpus.

When a record also carries the neural model's ``predicted_candidate``, we
additionally measure how often the neural answer agrees with the oracle.
"""

from __future__ import annotations

from typing import Any


def _empty_set_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "oracle_exact": 0,
        "oracle_exact_rate": 0.0,
        "agreement_count": 0,
        "agreement_rate": None,
    }


def _finalize_summary(
    count: int,
    oracle_exact: int,
    agreement_eligible: int,
    agreement_count: int,
) -> dict[str, Any]:
    """Turn running tallies into a finished per-set/overall summary block."""
    oracle_exact_rate = oracle_exact / count if count else 0.0
    agreement_rate = (
        agreement_count / agreement_eligible if agreement_eligible else None
    )
    return {
        "count": count,
        "oracle_exact": oracle_exact,
        "oracle_exact_rate": oracle_exact_rate,
        "agreement_count": agreement_count,
        "agreement_rate": agreement_rate,
    }


def oracle_benchmark(records_by_set: dict, responder) -> dict:
    """Benchmark eval records against the closed-world oracle.

    Args:
        records_by_set: mapping of set name -> list of record dicts. Each record
            has ``"prompt"`` (str), ``"target"`` (str), and optionally
            ``"predicted_candidate"`` (str, the neural model's answer).
        responder: any object exposing ``answer_prompt(prompt: str) -> str``
            (e.g. a :class:`~corpus_responder.CorpusResponder`). Dependency
            injected so callers/tests can supply a real oracle or a stub.

    Returns:
        ``{"per_set": {name: summary}, "overall": summary}`` where each summary
        is ``{"count", "oracle_exact", "oracle_exact_rate", "agreement_count",
        "agreement_rate"}``. ``agreement_rate`` is ``None`` when no record in
        scope carried a ``predicted_candidate``.
    """
    per_set: dict[str, Any] = {}

    overall_count = 0
    overall_oracle_exact = 0
    overall_agreement_eligible = 0
    overall_agreement_count = 0

    for set_name, records in records_by_set.items():
        count = 0
        oracle_exact = 0
        agreement_eligible = 0
        agreement_count = 0

        for record in records:
            oracle_answer = responder.answer_prompt(record["prompt"])
            oracle_correct = oracle_answer == record["target"]

            count += 1
            if oracle_correct:
                oracle_exact += 1

            if "predicted_candidate" in record:
                agreement_eligible += 1
                if record["predicted_candidate"] == oracle_answer:
                    agreement_count += 1

        per_set[set_name] = _finalize_summary(
            count, oracle_exact, agreement_eligible, agreement_count
        )

        overall_count += count
        overall_oracle_exact += oracle_exact
        overall_agreement_eligible += agreement_eligible
        overall_agreement_count += agreement_count

    overall = _finalize_summary(
        overall_count,
        overall_oracle_exact,
        overall_agreement_eligible,
        overall_agreement_count,
    )

    return {"per_set": per_set, "overall": overall}
