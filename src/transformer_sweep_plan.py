"""Controlled sweep-plan artifacts for transformer screens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformer_direct_answer_repair_targets import direct_answer_repair_target_profiles
from transformer_model import (
    optimization_config_from_args,
    transformer_config_from_args,
)


SWEEP_PLAN_KIND = "transformer_sweep_plan"
SWEEP_PLAN_SCHEMA_VERSION = 1


def build_transformer_sweep_plan(
    args: Any,
    tokenizer: Any,
    *,
    recipe_id: str,
) -> dict[str, Any]:
    trial = _trial_config(args, tokenizer)
    axes = {key: [value] for key, value in trial.items()}
    return {
        "schema_version": SWEEP_PLAN_SCHEMA_VERSION,
        "kind": SWEEP_PLAN_KIND,
        "component": "transformer-answer-train",
        "run_id": args.run.name,
        "recipe_id": recipe_id,
        "status": "single_controlled_trial_recorded",
        "rule": (
            "Transformer changes must be compared as controlled trials across "
            "declared tokenizer, architecture, optimizer, and epoch-budget axes "
            "instead of undocumented knob changes."
        ),
        "axes": axes,
        "current_trial": trial,
        "promotion_scope": "evidence_only_until_constraint_first_gate_passes",
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
    }


def sweep_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": plan.get("kind"),
        "status": plan.get("status"),
        "axis_count": len(plan.get("axes", {})),
        "tokenizer_type": plan.get("current_trial", {}).get("tokenizer_type"),
        "transformer_profile": plan.get("current_trial", {}).get("transformer_profile"),
        "direct_answer_mode": plan.get("current_trial", {}).get("direct_answer_mode"),
        "direct_answer_repair_target_profiles": plan.get("current_trial", {}).get(
            "direct_answer_repair_target_profiles"
        ),
        "direct_answer_frontier_metrics_path": plan.get("current_trial", {}).get(
            "direct_answer_frontier_metrics_path"
        ),
        "promotion_scope": plan.get("promotion_scope"),
    }


def write_transformer_sweep_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _trial_config(args: Any, tokenizer: Any) -> dict[str, Any]:
    model_config = transformer_config_from_args(args, tokenizer.vocab_size)
    optimizer_config = optimization_config_from_args(args)
    repair_target_profiles = direct_answer_repair_target_profiles(args)
    return {
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_manifest_hash": getattr(args, "tokenizer_manifest_hash", None),
        "transformer_profile": model_config.transformer_profile,
        "context_size": model_config.context_size,
        "embedding_dim": model_config.embedding_dim,
        "attention_heads": model_config.attention_heads,
        "num_layers": model_config.num_layers,
        "feedforward_dim": model_config.feedforward_dim,
        "use_pre_layer_norm": model_config.use_pre_layer_norm,
        "use_rms_norm": model_config.use_rms_norm,
        "use_gated_mlp": model_config.use_gated_mlp,
        "use_rotary_positions": model_config.use_rotary_positions,
        "optimizer": optimizer_config.optimizer,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "direct_answer_steps": args.direct_answer_steps,
        "direct_answer_mode": args.direct_answer_mode,
        "direct_answer_repair_target_profiles": repair_target_profiles,
        "direct_answer_learning_rate": args.direct_answer_learning_rate,
        "direct_answer_frontier_metrics_path": _frontier_metrics_path(args),
        "gradient_clip": optimizer_config.gradient_clip,
        "warmup_steps": optimizer_config.warmup_steps,
        "decay_steps": optimizer_config.decay_steps,
        "gradient_accumulation_steps": optimizer_config.gradient_accumulation_steps,
    }


def _frontier_metrics_path(args: Any) -> str | None:
    path = getattr(args, "direct_answer_frontier_metrics", None)
    return str(path) if path is not None else None
