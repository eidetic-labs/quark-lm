"""Acceptance-gate construction for transformer experiments."""

from __future__ import annotations

from typing import Any

from transformer_profile_scale_experiment_gates import profile_scale_experiment_gates


def parse_experiment_gate(raw_gate: str) -> dict[str, Any]:
    if ":" in raw_gate:
        name, rule = raw_gate.split(":", 1)
    else:
        name = raw_gate
        rule = raw_gate
    name = name.strip().replace(" ", "_")
    rule = rule.strip()
    if not name or not rule:
        raise ValueError("experiment gates must be formatted as name:rule")
    return {"name": name, "rule": rule, "required": True}


def transformer_experiment_acceptance_gates(args: Any) -> list[dict[str, Any]]:
    gates = [
        _required_gate(
            "baseline_snapshot_recorded",
            "Record a step-0 baseline before training updates.",
        ),
        _required_gate(
            "final_snapshot_recorded",
            "Record a final snapshot after training updates.",
        ),
        _required_gate(
            "closed_world_training_data",
            "Use only corpus-derived AnswerExample lessons and declared eval probes.",
        ),
        _required_gate(
            "training_recipe",
            "A recipe artifact must bind model, data, objective, optimizer, artifacts, and gates.",
        ),
        _required_gate(
            "closed_world_verifier",
            "Training plan, hygiene, and candidate quarantine checks must pass before training.",
        ),
        _required_gate(
            "controlled_sweep_plan",
            "Record tokenizer, architecture, optimizer, and training-budget sweep axes.",
        ),
        _required_gate(
            "replay_mixture_report",
            "Record new lessons, retained facts, unknown policy, tokenizer stress, and heldout/paraphrase mixture evidence.",
        ),
        _required_gate(
            "constraint_first_promotion",
            "Loss, NLL, rank, and top-k evidence may influence promotion only after constraints pass.",
        ),
        _required_gate(
            "no_pretrained_weights",
            (
                "Initialize transformer weights randomly or from declared "
                "QuarkLM checkpoints only."
            ),
        ),
        _required_gate(
            "no_pretrained_tokenizer",
            "Train the tokenizer from admitted corpus text.",
        ),
        _required_gate(
            "no_external_embeddings",
            "Do not import external embeddings or pretrained representation tables.",
        ),
        _required_gate(
            "backend_policy_recorded",
            (
                "Record scalar-reference or PyTorch-experimental backend metadata "
                "with purity flags and parity status."
            ),
        ),
    ]
    if getattr(args, "direct_answer_steps", 0) > 0:
        gates.extend(_direct_answer_gates())
        gates.extend(
            profile_scale_experiment_gates(getattr(args, "direct_answer_mode", ""))
        )
    gates.extend(
        parse_experiment_gate(raw_gate)
        for raw_gate in (getattr(args, "experiment_acceptance_gate", None) or [])
    )
    return gates


def _direct_answer_gates() -> list[dict[str, Any]]:
    return [
        _required_gate(
            "branch_context_gate_recorded",
            "Record semantic branch-context coverage for direct-answer screens.",
        ),
        _required_gate(
            "branch_diversity_recorded",
            "Record branch diversity for each direct-answer snapshot.",
        ),
        _required_gate(
            "target_coverage_recorded",
            "Record target coverage by branch profile.",
        ),
    ]


def _required_gate(name: str, rule: str) -> dict[str, Any]:
    return {"name": name, "rule": rule, "required": True}
