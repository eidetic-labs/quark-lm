"""Experiment-intent records for transformer answer runs."""

from __future__ import annotations

from typing import Any

from answer_model import DEFAULT_EVALS as DEFAULT_ANSWER_EVALS
from experiment_registry import ExperimentIntent
from transformer_experiment_constants import TRANSFORMER_RECIPE_VERSION
from transformer_experiment_gates import transformer_experiment_acceptance_gates
from transformer_experiment_modes import direct_answer_is_profile_aware
from transformer_experiment_recipe import transformer_training_recipe_id
from transformer_run_artifacts import TransformerRunArtifacts


def transformer_experiment_intent(args: Any) -> dict[str, Any]:
    hypothesis = getattr(args, "experiment_hypothesis", None) or (
        "A tiny decoder-only transformer can improve corpus-derived answer "
        "evidence under declared gates without leaving the closed-world data boundary."
    )
    notes = list(getattr(args, "experiment_note", None) or [])
    notes.append("Eval probe paths are declared for measurement, not as hidden training data.")
    failure_criteria = [
        "Metrics omit baseline or final snapshots.",
        "The run uses pretrained weights, pretrained tokenizers, or external embeddings.",
        "Direct-answer screens omit branch-context, branch-diversity, or target-coverage evidence.",
        "A screen writes checkpoints without experiment intent and metrics artifacts.",
        "A screen writes checkpoints without a matching training recipe artifact.",
        "A screen omits controlled sweep axes or replay-mixture evidence.",
        "The deterministic closed-world verifier rejects the training plan.",
        "The constraint-first promotion gate rejects the run before quality metrics are eligible.",
    ]
    failure_criteria.extend(getattr(args, "experiment_failure_criterion", None) or [])
    artifacts = TransformerRunArtifacts.from_run(
        args.run,
        direct_profile_aware=direct_answer_is_profile_aware(args),
    )
    intent = ExperimentIntent(
        version=getattr(args, "experiment_version", TRANSFORMER_RECIPE_VERSION),
        run_id=args.run.name,
        component="transformer-answer-train",
        hypothesis=hypothesis,
        allowed_data_sources=[
            str(args.train_text),
            str(args.valid),
            str(args.corpus_dir),
            *[str(path) for path in DEFAULT_ANSWER_EVALS],
        ],
        planned_artifacts=artifacts.intent_artifacts(),
        training_recipe_id=transformer_training_recipe_id(args),
        acceptance_gates=transformer_experiment_acceptance_gates(args),
        failure_criteria=failure_criteria,
        replay_plan_id=(
            "direct_answer_replay_plan.json"
            if direct_answer_is_profile_aware(args)
            else None
        ),
        notes=notes,
    )
    return intent.to_record()
