"""Plan gated weight consolidation from retrieval memory and neural failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REPORT_KIND = "memory_consolidation_plan"


def profile_priority_score(
    retrieval_summary: dict[str, Any],
    branch_profile: dict[str, Any],
) -> float:
    retrieval_exact_rate = float(retrieval_summary.get("exact_rate", 0.0))
    retrieved = int(retrieval_summary.get("retrieved", 0))
    target_coverage = float(branch_profile.get("target_token_coverage", 0.0))
    predicted_unique = int(branch_profile.get("predicted_unique", 0))
    target_unique = int(branch_profile.get("target_unique", 0))
    collapsed_bonus = 2.0 if branch_profile.get("collapsed") is True else 0.0
    coverage_gap = max(0.0, 1.0 - target_coverage)
    diversity_gap = max(0, target_unique - predicted_unique)
    retrieval_bonus = 1.0 if retrieval_exact_rate == 1.0 and retrieved > 0 else 0.0
    focus_bonus = 0.5 if branch_profile.get("name") in {"owner", "paraphrases"} else 0.0
    return round(
        retrieval_bonus + collapsed_bonus + coverage_gap + (diversity_gap * 0.1) + focus_bonus,
        6,
    )


def memory_cards_for_profile(retrieval_summary: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for record in retrieval_summary.get("records", []):
        if not isinstance(record, dict) or not record.get("retrieved"):
            continue
        card_id = record.get("memory_card_id")
        if not isinstance(card_id, str) or not card_id:
            continue
        cards.setdefault(
            card_id,
            {
                "memory_card_id": card_id,
                "memory_card_source": record.get("memory_card_source"),
                "record_ids": [],
                "targets": [],
            },
        )
        cards[card_id]["record_ids"].append(record.get("id"))
        target = record.get("target")
        if target not in cards[card_id]["targets"]:
            cards[card_id]["targets"].append(target)
    return sorted(cards.values(), key=lambda item: item["memory_card_id"])[:limit]


def build_memory_consolidation_plan(
    retrieval_report: dict[str, Any],
    transformer_metrics: dict[str, Any],
) -> dict[str, Any]:
    direct_answer = transformer_metrics.get("direct_answer", {})
    final_snapshot = direct_answer.get("final", {}) if isinstance(direct_answer, dict) else {}
    branch_target = (
        final_snapshot.get("branch_diversity_target", {})
        if isinstance(final_snapshot, dict)
        else {}
    )
    blocking_profiles = [
        profile
        for profile in branch_target.get("blocking_evals", [])
        if isinstance(profile, dict)
    ]
    retrieval_evals = retrieval_report.get("evals", {})
    profile_priorities: list[dict[str, Any]] = []
    for branch_profile in blocking_profiles:
        name = branch_profile.get("name")
        retrieval_summary = (
            retrieval_evals.get(name, {})
            if isinstance(name, str) and isinstance(retrieval_evals, dict)
            else {}
        )
        retrieved = int(retrieval_summary.get("retrieved", 0) or 0)
        retrieval_exact_rate = float(retrieval_summary.get("exact_rate", 0.0) or 0.0)
        memory_backed = retrieval_exact_rate == 1.0 and retrieved > 0
        if not memory_backed:
            continue
        profile_priorities.append(
            {
                "profile": name,
                "priority_score": profile_priority_score(retrieval_summary, branch_profile),
                "retrieval_exact_rate": retrieval_exact_rate,
                "retrieval_record_count": int(retrieval_summary.get("count", 0) or 0),
                "retrieved_records": retrieved,
                "neural_target_token_coverage": float(
                    branch_profile.get("target_token_coverage", 0.0) or 0.0
                ),
                "neural_predicted_unique": int(branch_profile.get("predicted_unique", 0) or 0),
                "target_unique": int(branch_profile.get("target_unique", 0) or 0),
                "collapsed": branch_profile.get("collapsed") is True,
                "dominant_predicted_token": branch_profile.get("dominant_predicted_token"),
                "dominant_predicted_rate": float(
                    branch_profile.get("dominant_predicted_rate", 0.0) or 0.0
                ),
                "missing_target_tokens": list(branch_profile.get("missing_target_tokens", [])),
                "memory_cards": memory_cards_for_profile(retrieval_summary),
                "recommended_action": (
                    "consolidate_retrieved_memory_with_branch_diversity_gate"
                ),
            }
        )
    profile_priorities.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            str(item["profile"]),
        )
    )
    collapsed_profiles = [
        item["profile"] for item in profile_priorities if item["collapsed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "dataset_exclusivity": {
            "uses_external_model": False,
            "external_embeddings": False,
            "pretrained_retriever": False,
            "updates_weights": False,
            "source": "retrieval_memory_report plus transformer branch diagnostics",
        },
        "inputs": {
            "retrieval_report_kind": retrieval_report.get("kind"),
            "transformer_run_id": transformer_metrics.get("run_id"),
            "transformer_metrics_path": transformer_metrics.get("metrics_path"),
        },
        "summary": {
            "retrieval_record_count": retrieval_report.get("summary", {}).get("record_count", 0),
            "retrieval_exact_rate": retrieval_report.get("summary", {}).get("exact_rate", 0.0),
            "branch_diversity_passed": branch_target.get("passed") is True,
            "neural_failed_profiles": int(branch_target.get("failed_profiles", 0) or 0),
            "memory_backed_failed_profiles": len(profile_priorities),
            "collapsed_memory_backed_profiles": collapsed_profiles,
            "top_priority_profiles": [
                item["profile"] for item in profile_priorities[:5]
            ],
        },
        "profile_priorities": profile_priorities,
        "self_improvement": {
            "status": "planned_memory_guided_weight_consolidation",
            "rule": (
                "Retrieval success identifies admitted memories available for immediate "
                "answering; weight consolidation remains pending until branch-diversity "
                "and target-token gates pass."
            ),
            "next_step": (
                "Train only memory-backed profiles whose neural branch predictions are "
                "collapsed or missing target tokens, preserving retrieval provenance."
            ),
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_memory_consolidation_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--transformer-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_memory_consolidation_plan(
        read_json(args.retrieval_report),
        read_json(args.transformer_metrics),
    )
    if args.output is not None:
        write_memory_consolidation_plan(args.output, plan)
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
