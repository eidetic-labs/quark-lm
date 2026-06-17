"""Branch-context coverage auditing for direct-answer evaluations."""

from __future__ import annotations

from collections import Counter
from typing import Any

from answer_model import AnswerExample, semantic_feature_names
from tokenizer import CharTokenizer
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_modes import ANSWER_TERMINATOR


def audit_direct_answer_branch_context_coverage(
    model: Any,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    max_records: int = 12,
) -> dict[str, Any]:
    semantic_records = 0
    covered = 0
    skipped = 0
    missing_records: list[dict[str, Any]] = []
    context_records: dict[str, list[dict[str, Any]]] = {}
    context_targets: dict[str, Counter[str]] = {}
    target_counts: Counter[str] = Counter()

    for record in records:
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source=f"eval:{record['id']}",
        )
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            skipped += 1
            continue
        context, target_id, position = branch
        context_text = tokenizer.decode(context)
        target_token = tokenizer.itos[target_id]
        target_counts[target_token] += 1
        context_records.setdefault(context_text, []).append(
            {
                "id": record["id"],
                "target": record["target"],
                "branch_position": position,
                "target_token": target_token,
            }
        )
        context_targets.setdefault(context_text, Counter())[target_token] += 1

        full_features = set(semantic_feature_names(record["prompt"].lower()))
        if not full_features:
            continue
        semantic_records += 1
        context_features = set(semantic_feature_names(context_text.lower()))
        missing_features = sorted(full_features - context_features)
        if not missing_features:
            covered += 1
            continue
        if len(missing_records) < max_records:
            missing_records.append(
                {
                    "id": record["id"],
                    "branch_position": position,
                    "context_size": model.config.context_size,
                    "context_text": context_text,
                    "missing_features": missing_features,
                    "target_token": target_token,
                }
            )

    ambiguous_records, context_stats = _context_ambiguity_records(
        context_records,
        context_targets,
        max_records,
    )
    count = sum(len(records_for_context) for records_for_context in context_records.values())
    return {
        "branch_position": branch_position,
        "context_size": model.config.context_size,
        "count": count,
        "skipped": skipped,
        "semantic_records": semantic_records,
        "covered": covered,
        "missing": semantic_records - covered,
        "covered_rate": covered / semantic_records if semantic_records else 1.0,
        "unique_contexts": len(context_records),
        "collision_contexts": context_stats["collision_contexts"],
        "ambiguous_contexts": context_stats["ambiguous_contexts"],
        "max_context_reuse": context_stats["max_context_reuse"],
        "max_target_options": context_stats["max_target_options"],
        "target_tokens": [
            {"value": value, "count": count}
            for value, count in target_counts.most_common(12)
        ],
        "missing_records": missing_records,
        "ambiguous_records": ambiguous_records,
    }


def _context_ambiguity_records(
    context_records: dict[str, list[dict[str, Any]]],
    context_targets: dict[str, Counter[str]],
    max_records: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ambiguous_records: list[dict[str, Any]] = []
    collision_contexts = 0
    ambiguous_contexts = 0
    max_context_reuse = 0
    max_target_options = 0
    for context_text, examples in context_records.items():
        target_counter = context_targets[context_text]
        max_context_reuse = max(max_context_reuse, len(examples))
        max_target_options = max(max_target_options, len(target_counter))
        if len(examples) > 1:
            collision_contexts += 1
        if len(target_counter) <= 1:
            continue
        ambiguous_contexts += 1
        if len(ambiguous_records) < max_records:
            ambiguous_records.append(
                {
                    "context_text": context_text,
                    "count": len(examples),
                    "target_tokens": [
                        {"value": value, "count": count}
                        for value, count in target_counter.most_common(12)
                    ],
                    "records": examples[:max_records],
                }
            )
    return ambiguous_records, {
        "collision_contexts": collision_contexts,
        "ambiguous_contexts": ambiguous_contexts,
        "max_context_reuse": max_context_reuse,
        "max_target_options": max_target_options,
    }


def summarize_branch_context_coverage_gate(
    coverage_by_eval: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total_count = 0
    semantic_records = 0
    covered = 0
    missing = 0
    ambiguous_contexts = 0
    collision_contexts = 0
    skipped = 0
    blocking_evals: list[dict[str, Any]] = []
    for name, coverage in sorted(coverage_by_eval.items()):
        total_count += coverage["count"]
        semantic_records += coverage["semantic_records"]
        covered += coverage["covered"]
        missing += coverage["missing"]
        ambiguous_contexts += coverage["ambiguous_contexts"]
        collision_contexts += coverage["collision_contexts"]
        skipped += coverage["skipped"]
        if coverage["missing"] or coverage["ambiguous_contexts"] or coverage["skipped"]:
            blocking_evals.append(
                {
                    "name": name,
                    "count": coverage["count"],
                    "missing": coverage["missing"],
                    "ambiguous_contexts": coverage["ambiguous_contexts"],
                    "skipped": coverage["skipped"],
                    "covered_rate": coverage["covered_rate"],
                }
            )
    passed = missing == 0 and ambiguous_contexts == 0 and skipped == 0
    return {
        "passed": passed,
        "count": total_count,
        "semantic_records": semantic_records,
        "covered": covered,
        "missing": missing,
        "covered_rate": covered / semantic_records if semantic_records else 1.0,
        "ambiguous_contexts": ambiguous_contexts,
        "collision_contexts": collision_contexts,
        "skipped": skipped,
        "blocking_evals": blocking_evals,
    }
