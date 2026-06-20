from __future__ import annotations

from typing import Any

from transformer_backend_policy import transformer_backend_metadata
from transformer_experiment import (
    PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
    TRAINING_DATA_DESCRIPTION,
)


def metrics(
    *,
    diversity_passed: bool,
    baseline_coverage: float,
    final_coverage: float,
) -> dict[str, Any]:
    return {
        "run_id": "bundle-a",
        "experiment_bundle": PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
        "baseline": {"step": 0},
        "final": {"step": 1},
        "training_data": TRAINING_DATA_DESCRIPTION,
        "closed_world_verifier": {"passed": True},
        "training_recipe": {"recipe_id": "transformer-answer:test"},
        "sweep_plan": {"kind": "transformer_sweep_plan"},
        "sweep_plan_path": "runs/bundle-a/sweep_plan.json",
        "replay_mixture_report": {"summary": {"passed": True}},
        "replay_mixture_report_path": "runs/bundle-a/replay_mixture_report.json",
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "backend": transformer_backend_metadata(seed=17, tokenizer_type="char"),
        "direct_answer": {
            "direct_answer_weight_update_outcome": {
                "status": "accepted",
                "accepted": True,
            },
            "direct_answer_branch_context_gate": {"passed": True},
            "routing_repair_batch_evidence": routing_repair_batch_evidence(),
            "baseline": snapshot(baseline_coverage, diversity_passed=True),
            "final": snapshot(final_coverage, diversity_passed=diversity_passed),
        },
    }


def rank_collapse_metrics(
    *,
    diversity_passed: bool,
    baseline_coverage: float,
    final_coverage: float,
) -> dict[str, Any]:
    payload = metrics(
        diversity_passed=diversity_passed,
        baseline_coverage=baseline_coverage,
        final_coverage=final_coverage,
    )
    payload["experiment_bundle"] = (
        PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE
    )
    direct_answer = payload["direct_answer"]
    direct_answer["direct_answer_mode"] = (
        "branch-profile-balanced-rank-collapse-unlikelihood"
    )
    direct_answer["direct_answer_contrast_weight"] = 1.0
    direct_answer["direct_answer_hard_negatives"] = 4
    return payload


def snapshot(coverage: float, *, diversity_passed: bool) -> dict[str, Any]:
    return {
        "branch_target_coverage_by_profile": {"qa": coverage},
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 2,
                    "target_token_coverage": coverage,
                }
            }
        },
        "branch_representation_profiles": {
            "qa": {
                "target_unique": 2,
                "target_centroid_distance": {"min": 0.2, "avg": 0.3},
                "target_centroid_margin": {
                    "min": 0.1,
                    "avg": 0.2,
                    "poorly_separated_rate": 0.0,
                },
            }
        },
        "branch_diversity_target": {
            "passed": diversity_passed,
            "multi_target_profiles": 1,
            "passed_profiles": 1 if diversity_passed else 0,
            "failed_profiles": 0 if diversity_passed else 1,
            "blocking_evals": [] if diversity_passed else [{"profile": "qa"}],
        },
        "branch_routing_audit": {
            "representation": {
                "profile_count": 1,
                "low_separation_profile_count": 0,
                "profiles": [{"profile": "qa"}],
            },
            "logit_prior": {
                "profile_count": 1,
                "hidden_projection_profile_count": 1,
                "mixed_profile_count": 0,
                "pressure_counts": {"hidden_projection": 1},
                "profiles": [{"profile": "qa"}],
            },
        },
        "evals": {"qa": {"count": 1, "exact": 1}},
    }


def routing_repair_batch_evidence() -> dict[str, Any]:
    return {
        "bundle": PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
        "batch_builder": "profile-balanced-training-family-branch-batch",
        "step_count": 1,
        "branch_count": 3,
        "profiles": ["glossary", "learning", "qa"],
        "required_eval_profiles": {
            "failed_profiles": ["glossary", "learning", "qa"],
            "zero_coverage_profiles": ["learning"],
            "buried_target_profiles": ["glossary", "qa"],
        },
        "profile_balanced_branch_batches": {
            "name": "profile_balanced_branch_batches",
            "passed": True,
            "status": "passed",
            "required_trainable_profiles": ["glossary", "learning", "qa"],
            "covered_trainable_profiles": ["glossary", "learning", "qa"],
            "missing_trainable_profiles": [],
            "eval_only_profiles": [],
        },
        "steps": [],
        "passed": True,
        "status": "passed",
    }
