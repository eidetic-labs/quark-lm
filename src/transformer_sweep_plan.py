"""Controlled sweep-plan artifacts for transformer screens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
        "promotion_scope": plan.get("promotion_scope"),
    }


def write_transformer_sweep_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _trial_config(args: Any, tokenizer: Any) -> dict[str, Any]:
    return {
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_manifest_hash": getattr(args, "tokenizer_manifest_hash", None),
        "transformer_profile": getattr(args, "transformer_profile", "default"),
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "attention_heads": args.attention_heads,
        "num_layers": args.num_layers,
        "feedforward_dim": args.feedforward_dim,
        "optimizer": getattr(args, "optimizer", "sgd"),
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "direct_answer_steps": args.direct_answer_steps,
        "direct_answer_mode": args.direct_answer_mode,
        "direct_answer_learning_rate": args.direct_answer_learning_rate,
        "gradient_clip": getattr(args, "gradient_clip", 0.0),
        "warmup_steps": getattr(args, "warmup_steps", 0),
        "decay_steps": getattr(args, "decay_steps", 0),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
    }
