"""Promotion-decision evidence for transformer answer experiments."""

from __future__ import annotations

from typing import Any

from transformer_experiment_constants import TRAINING_DATA_DESCRIPTION


def transformer_experiment_decision(
    metrics: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    constraint_gate = metrics.get("constraint_first_promotion", {})
    evidence = [
        {"name": "baseline_snapshot_recorded", "passed": bool(metrics.get("baseline"))},
        {"name": "final_snapshot_recorded", "passed": bool(metrics.get("final"))},
        {
            "name": "closed_world_training_data",
            "passed": metrics.get("training_data") == TRAINING_DATA_DESCRIPTION,
        },
        {
            "name": "closed_world_verifier",
            "passed": metrics.get("closed_world_verifier", {}).get("passed") is True,
        },
        {
            "name": "training_recipe",
            "passed": "training_recipe" in metrics,
        },
        {
            "name": "controlled_sweep_plan",
            "passed": metrics.get("sweep_plan", {}).get("kind")
            == "transformer_sweep_plan",
        },
        {
            "name": "replay_mixture_report",
            "passed": metrics.get("replay_mixture_report", {})
            .get("summary", {})
            .get("passed")
            is True,
        },
        {
            "name": "constraint_first_promotion",
            "passed": constraint_gate.get("passed") is True,
            "status": constraint_gate.get("status"),
        },
        {
            "name": "no_pretrained_weights",
            "passed": metrics.get("pretrained_weights") is False,
        },
        {
            "name": "no_pretrained_tokenizer",
            "passed": metrics.get("pretrained_tokenizer") is False,
        },
        {
            "name": "no_external_embeddings",
            "passed": metrics.get("external_embeddings") is False,
        },
    ]
    _append_direct_answer_evidence(evidence, metrics.get("direct_answer"))
    if constraint_gate.get("passed") is True:
        return (
            "promoted",
            "Transformer run passed the constraint-first promotion gate.",
            evidence,
        )
    return (
        "rejected",
        (
            "Transformer run rejected by the constraint-first promotion gate; "
            "quality metrics cannot promote unless constraints pass first."
        ),
        evidence,
    )


def _append_direct_answer_evidence(
    evidence: list[dict[str, Any]],
    direct_answer: Any,
) -> None:
    if not isinstance(direct_answer, dict):
        return
    branch_gate = direct_answer.get("direct_answer_branch_context_gate")
    final_snapshot = direct_answer.get("final", {})
    diversity = final_snapshot.get("branch_diversity_target", {})
    coverage = final_snapshot.get("branch_target_coverage_by_profile", {})
    evidence.extend(
        [
            {
                "name": "branch_context_gate_recorded",
                "passed": isinstance(branch_gate, dict),
            },
            {
                "name": "branch_diversity_recorded",
                "passed": isinstance(diversity, dict),
            },
            {
                "name": "target_coverage_recorded",
                "passed": isinstance(coverage, dict),
            },
        ]
    )
    if isinstance(branch_gate, dict):
        evidence.append(
            {
                "name": "branch_context_gate",
                "passed": bool(branch_gate.get("passed")),
            }
        )
    if isinstance(diversity, dict):
        evidence.append(
            {
                "name": "branch_diversity_target",
                "passed": bool(diversity.get("passed")),
            }
        )
