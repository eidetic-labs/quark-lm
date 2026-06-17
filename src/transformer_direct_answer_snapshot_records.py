"""Direct-answer snapshot record construction."""

from __future__ import annotations

from typing import Any

from branch_diversity_diagnostics import branch_routing_audit_summary
from branch_diversity_snapshot_coverage import branch_diversity_snapshot_target_coverage_by_profile
from tokenizer import CharTokenizer
from transformer_branch_logit_diagnostics import (
    direct_answer_branch_logit_prior_profile,
    summarize_branch_diversity_target,
)
from transformer_branch_profiles import (
    direct_answer_branch_profile,
    direct_answer_branch_representation_profile,
)
from transformer_direct_answer_branch_context_evaluation import (
    audit_direct_answer_branch_context_coverage,
    summarize_branch_context_coverage_gate,
)
from transformer_direct_answer_evaluation import evaluate_direct_answer_records
from transformer_model import GenerationConfig


def direct_answer_snapshot_record(
    model: Any,
    tokenizer: CharTokenizer,
    eval_records: dict[str, list[dict[str, Any]]],
    branch_position: int,
    max_new_chars: int,
    snapshot_mode: str,
    terminator: str,
    generation_config: GenerationConfig,
    step: int,
    train_loss: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direct_answer_evals_skipped = snapshot_mode == "branch-only"
    coverage_only_probe = bool(
        extra
        and extra.get("baseline_floor_update_guard_probe")
        and extra.get("baseline_floor_coverage_only_probe", True)
    )
    branch_context_coverage = (
        {}
        if coverage_only_probe
        else {
            name: audit_direct_answer_branch_context_coverage(
                model,
                tokenizer,
                records,
                branch_position,
                terminator,
            )
            for name, records in sorted(eval_records.items())
        }
    )
    branch_profiles = {
        name: direct_answer_branch_profile(
            model,
            tokenizer,
            records,
            branch_position,
            terminator,
        )
        for name, records in sorted(eval_records.items())
    }
    branch_representation_profiles = (
        {}
        if coverage_only_probe
        else {
            name: direct_answer_branch_representation_profile(
                model,
                tokenizer,
                records,
                branch_position,
                terminator,
            )
            for name, records in sorted(eval_records.items())
        }
    )
    branch_logit_prior_profiles = {
        name: direct_answer_branch_logit_prior_profile(
            model,
            tokenizer,
            records,
            branch_position,
            terminator,
        )
        for name, records in sorted(eval_records.items())
    }
    record = {
        "step": step,
        "train_loss": train_loss,
        "direct_answer_snapshot_mode": snapshot_mode,
        "evals_skipped": direct_answer_evals_skipped,
        "evals": {}
        if direct_answer_evals_skipped
        else {
            name: evaluate_direct_answer_records(
                model,
                tokenizer,
                records,
                max_new_chars,
                terminator,
                generation_config,
            )
            for name, records in sorted(eval_records.items())
        },
        "branch_profiles": branch_profiles,
        "branch_representation_profiles": branch_representation_profiles,
        "branch_logit_prior_profiles": branch_logit_prior_profiles,
        "branch_diversity_target": summarize_branch_diversity_target(branch_profiles),
        "branch_routing_audit": branch_routing_audit_summary(
            branch_profiles,
            branch_representation_profiles,
            {
                tokenizer.itos[index]: value.data
                for index, value in enumerate(model.bout)
            },
            branch_logit_prior_profiles,
        ),
        "branch_context_coverage": branch_context_coverage,
        "branch_context_gate": summarize_branch_context_coverage_gate(
            branch_context_coverage
        ),
        "coverage_only_probe": coverage_only_probe,
    }
    record["branch_target_coverage_by_profile"] = (
        branch_diversity_snapshot_target_coverage_by_profile(record)
    )
    if extra is not None:
        record.update(extra)
    return record
