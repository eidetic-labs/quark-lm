"""Transformer experiment, recipe, and promotion-decision surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_model import DEFAULT_EVALS as DEFAULT_ANSWER_EVALS
from .experiment_registry import ExperimentIntent
from .training_recipe import build_training_recipe
from .transformer_model import TRANSFORMER_ARCHITECTURE, TRANSFORMER_TOKENIZER


TRANSFORMER_RECIPE_VERSION = "v0.78"
TRAINING_DATA_DESCRIPTION = (
    "closed_world_lm.answer_model corpus-derived AnswerExample lessons"
)
PROFILE_SCALE_DIVERSITY_MODE = (
    "branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-"
    "sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COVERAGE_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-prep-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-recovery-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-frontier-profile-scale-calibrated-sequential-"
    "profile-stabilization-unlikelihood"
)
PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "frontier-profile-scale-calibrated-sequential-profile-stabilization-"
    "unlikelihood"
)
PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
PROFILE_AWARE_DIRECT_ANSWER_MODES = {
    "branch-context-profile-coverage-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-anchored-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-target-share-preserving-deficit-unlikelihood",
    "branch-context-profile-baseline-floor-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    PROFILE_SCALE_DIVERSITY_MODE,
    PROFILE_SCALE_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE,
}


@dataclass(frozen=True)
class TransformerRunArtifacts:
    checkpoint: Path
    optimizer_state: Path
    tokenizer: Path
    corpus_hygiene: Path
    training_plan: Path
    training_recipe: Path
    candidate_quarantine: Path
    closed_world_verifier: Path
    constraint_first_promotion: Path
    retrieval_memory: Path
    memory_consolidation_plan: Path
    metrics: Path
    metrics_history: Path
    lessons: Path
    experiment_intent: Path
    replay_plan: Path | None = None

    @classmethod
    def from_run(
        cls,
        run_dir: Path,
        direct_profile_aware: bool = False,
    ) -> "TransformerRunArtifacts":
        return cls(
            checkpoint=run_dir / "transformer_answer.json",
            optimizer_state=run_dir / "optimizer_state.json",
            tokenizer=run_dir / "tokenizer.json",
            corpus_hygiene=run_dir / "corpus_hygiene.json",
            training_plan=run_dir / "training_plan.json",
            training_recipe=run_dir / "training_recipe.json",
            candidate_quarantine=run_dir / "candidate_quarantine.json",
            closed_world_verifier=run_dir / "closed_world_verifier.json",
            constraint_first_promotion=run_dir / "constraint_first_promotion.json",
            retrieval_memory=run_dir / "retrieval_memory_report.json",
            memory_consolidation_plan=run_dir / "memory_consolidation_plan.json",
            metrics=run_dir / "transformer_answer_metrics.json",
            metrics_history=run_dir / "transformer_answer_metrics.jsonl",
            lessons=run_dir / "transformer_answer_lessons.jsonl",
            experiment_intent=run_dir / "experiment_intent.json",
            replay_plan=(
                run_dir / "direct_answer_replay_plan.json"
                if direct_profile_aware
                else None
            ),
        )

    def training_plan_artifacts(self) -> list[Path]:
        return [
            self.checkpoint,
            self.optimizer_state,
            self.tokenizer,
            self.corpus_hygiene,
            self.training_plan,
            self.training_recipe,
            self.candidate_quarantine,
            self.closed_world_verifier,
            self.constraint_first_promotion,
            self.retrieval_memory,
            self.memory_consolidation_plan,
            self.metrics,
        ]

    def intent_artifacts(self) -> list[str]:
        paths = [
            self.checkpoint,
            self.optimizer_state,
            self.tokenizer,
            self.corpus_hygiene,
            self.training_plan,
            self.training_recipe,
            self.candidate_quarantine,
            self.closed_world_verifier,
            self.constraint_first_promotion,
            self.retrieval_memory,
            self.memory_consolidation_plan,
            self.metrics,
            self.metrics_history,
            self.lessons,
            self.experiment_intent,
        ]
        if self.replay_plan is not None:
            paths.append(self.replay_plan)
        return [str(path) for path in paths]


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


def is_profile_aware_direct_answer_mode(mode: str) -> bool:
    return mode in PROFILE_AWARE_DIRECT_ANSWER_MODES


def direct_answer_is_profile_aware(args: Any) -> bool:
    return (
        getattr(args, "direct_answer_steps", 0) > 0
        and is_profile_aware_direct_answer_mode(
            getattr(args, "direct_answer_mode", "")
        )
    )


def transformer_training_recipe_id(args: Any) -> str:
    mode = (
        getattr(args, "direct_answer_mode", "target-loss")
        if getattr(args, "direct_answer_steps", 0) > 0
        else "target-loss"
    )
    return f"transformer-answer:{mode}:{TRANSFORMER_RECIPE_VERSION}"


def transformer_experiment_acceptance_gates(args: Any) -> list[dict[str, Any]]:
    gates = [
        {
            "name": "baseline_snapshot_recorded",
            "rule": "Record a step-0 baseline before training updates.",
            "required": True,
        },
        {
            "name": "final_snapshot_recorded",
            "rule": "Record a final snapshot after training updates.",
            "required": True,
        },
        {
            "name": "closed_world_training_data",
            "rule": "Use only corpus-derived AnswerExample lessons and declared eval probes.",
            "required": True,
        },
        {
            "name": "training_recipe",
            "rule": "A recipe artifact must bind model, data, objective, optimizer, artifacts, and gates.",
            "required": True,
        },
        {
            "name": "closed_world_verifier",
            "rule": "Training plan, hygiene, and candidate quarantine checks must pass before training.",
            "required": True,
        },
        {
            "name": "constraint_first_promotion",
            "rule": "Loss, NLL, rank, and top-k evidence may influence promotion only after constraints pass.",
            "required": True,
        },
        {
            "name": "no_pretrained_weights",
            "rule": (
                "Initialize transformer weights randomly or from declared "
                "QuarkLM checkpoints only."
            ),
            "required": True,
        },
        {
            "name": "no_pretrained_tokenizer",
            "rule": "Train the tokenizer from admitted corpus text.",
            "required": True,
        },
        {
            "name": "no_external_embeddings",
            "rule": "Do not import external embeddings or pretrained representation tables.",
            "required": True,
        },
    ]
    if getattr(args, "direct_answer_steps", 0) > 0:
        gates.extend(
            [
                {
                    "name": "branch_context_gate_recorded",
                    "rule": "Record semantic branch-context coverage for direct-answer screens.",
                    "required": True,
                },
                {
                    "name": "branch_diversity_recorded",
                    "rule": "Record branch diversity for each direct-answer snapshot.",
                    "required": True,
                },
                {
                    "name": "target_coverage_recorded",
                    "rule": "Record target coverage by branch profile.",
                    "required": True,
                },
            ]
        )
        if getattr(args, "direct_answer_mode", "") == PROFILE_SCALE_DIVERSITY_MODE:
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_diversity_calibrated_"
                        "sequential_stabilization_screen"
                    ),
                    "rule": (
                        "Run records diversity-aware profile-scale search "
                        "activation, outer/search scales, source-profile scale "
                        "attempts, accepted/rejected diversity outcome counts, "
                        "score regression rejections, floor regression "
                        "rejections, accepted profile scales, update-shape "
                        "counts, replay plan, branch-context gate, coverage "
                        "floor, diversity target, recipe, verifier, and "
                        "constraint-first promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if getattr(args, "direct_answer_mode", "") == PROFILE_SCALE_FRONTIER_MODE:
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_frontier_calibrated_"
                        "sequential_stabilization_screen"
                    ),
                    "rule": (
                        "Run records frontier target-anchor search activation, "
                        "outer/search scales, source-profile scale attempts, "
                        "frontier anchor counts, accepted/rejected diversity "
                        "outcome counts, score regression rejections, floor "
                        "regression rejections, accepted profile scales, "
                        "update-shape counts, replay plan, branch-context gate, "
                        "coverage floor, diversity target, recipe, verifier, "
                        "and constraint-first promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_COVERAGE_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_coverage_frontier_"
                        "calibrated_sequential_stabilization_screen"
                    ),
                    "rule": (
                        "Run records coverage-frontier target-anchor search "
                        "activation, outer/search scales, source-profile scale "
                        "attempts, frontier anchor counts, coverage gain/tie/"
                        "regression counts, coverage rejection reasons, "
                        "accepted coverage deltas, diversity outcome counts, "
                        "floor regression rejections, accepted profile scales, "
                        "update-shape counts, replay plan, branch-context gate, "
                        "coverage floor, diversity target, recipe, verifier, "
                        "and constraint-first promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_coverage_prep_frontier_"
                        "calibrated_sequential_stabilization_screen"
                    ),
                    "rule": (
                        "Run records coverage-prep frontier activation, "
                        "outer/search scales, source-profile scale attempts, "
                        "frontier anchor counts, coverage gain/tie/regression "
                        "counts, coverage-preparation acceptances, coverage "
                        "rejection reasons, accepted preparation outcomes, "
                        "diversity outcome counts, floor regression "
                        "rejections, accepted profile scales, update-shape "
                        "counts, replay plan, branch-context gate, coverage "
                        "floor, diversity target, recipe, verifier, and "
                        "constraint-first promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_coverage_recovery_frontier_"
                        "calibrated_sequential_stabilization_screen"
                    ),
                    "rule": (
                        "Run records coverage-recovery frontier activation, "
                        "outer/search scales, source-profile scale attempts, "
                        "frontier anchor counts, coverage-preparation "
                        "candidates, recovery retry scales, recovery attempts, "
                        "recovery acceptances, fallback preparations, recovery "
                        "rejection reasons, accepted recovery outcomes, "
                        "coverage gain/tie/regression counts, diversity "
                        "outcome counts, floor regression rejections, accepted "
                        "profile scales, update-shape counts, replay plan, "
                        "branch-context gate, coverage floor, diversity "
                        "target, recipe, verifier, and constraint-first "
                        "promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_branch_stable_"
                        "coverage_recovery_frontier_calibrated_sequential_"
                        "stabilization_screen"
                    ),
                    "rule": (
                        "Run records branch-stable coverage-recovery frontier "
                        "activation, outer/search scales, source-profile "
                        "scale attempts, frontier anchor counts, "
                        "coverage-preparation candidates, recovery retry "
                        "scales, recovery attempts, branch-stability checks, "
                        "branch-stable recovery acceptances, fallback "
                        "preparations, branch-stability rejection reasons, "
                        "accepted branch-stable outcomes, coverage "
                        "gain/tie/regression counts, diversity outcome "
                        "counts, floor regression rejections, accepted "
                        "profile scales, update-shape counts, replay plan, "
                        "branch-context gate, coverage floor, diversity "
                        "target, recipe, verifier, and constraint-first "
                        "promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_branch_diversity_"
                        "recovery_frontier_calibrated_sequential_"
                        "stabilization_screen"
                    ),
                    "rule": (
                        "Run records branch-diversity recovery frontier "
                        "activation, outer/search scales, source-profile "
                        "scale attempts, frontier anchor counts, "
                        "coverage-preparation candidates, recovery retry "
                        "scales, branch-stability checks, branch-stable "
                        "recovery acceptances, branch-diversity recovery "
                        "candidates, attempts, acceptances, fallback "
                        "acceptances, rejection reasons, score deltas, "
                        "accepted branch-diversity outcomes, coverage "
                        "gain/tie/regression counts, diversity outcome "
                        "counts, floor regression rejections, accepted "
                        "profile scales, update-shape counts, replay plan, "
                        "branch-context gate, coverage floor, diversity "
                        "target, recipe, verifier, and constraint-first "
                        "promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_collapsed_profile_"
                        "binding_frontier_calibrated_sequential_"
                        "stabilization_screen"
                    ),
                    "rule": (
                        "Run records collapsed-profile binding frontier "
                        "activation, outer/search scales, source-profile "
                        "scale attempts, frontier anchor counts, "
                        "coverage-preparation candidates, recovery retry "
                        "scales, branch-stability checks, branch-stable "
                        "recovery acceptances, branch-diversity recovery "
                        "candidates, collapsed-profile binding candidates, "
                        "attempts, acceptances, fallback acceptances, "
                        "rejection reasons, target collapsed profiles, "
                        "profile-diversity deltas, score deltas, coverage "
                        "gain/tie/regression counts, diversity outcome "
                        "counts, floor regression rejections, accepted "
                        "profile scales, update-shape counts, replay plan, "
                        "branch-context gate, coverage floor, diversity "
                        "target, recipe, verifier, and constraint-first "
                        "promotion artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_remaining_profile_"
                        "binding_frontier_calibrated_sequential_"
                        "stabilization_screen"
                    ),
                    "rule": (
                        "Run records remaining-profile binding frontier "
                        "activation, target eval profiles, prioritized source "
                        "labels and source profiles, prioritized attempts, "
                        "prioritized acceptances and rejections, collapsed-"
                        "profile binding candidates, attempts, acceptances, "
                        "fallback acceptances, rejection reasons, profile-"
                        "diversity deltas, update-shape counts, replay plan, "
                        "branch-context gate, coverage floor, diversity target, "
                        "recipe, verifier, and constraint-first promotion "
                        "artifacts."
                    ),
                    "required": True,
                }
            )
        if (
            getattr(args, "direct_answer_mode", "")
            == PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE
        ):
            gates.append(
                {
                    "name": (
                        "baseline_floor_profile_scale_owner_paraphrase_"
                        "binding_frontier_calibrated_sequential_"
                        "stabilization_screen"
                    ),
                    "rule": (
                        "Run records owner/paraphrase binding frontier "
                        "activation, residual target eval profiles, preserved "
                        "learning profile, prioritized source labels and source "
                        "profiles, prioritized attempts, prioritized "
                        "acceptances and rejections, preservation checks, "
                        "preservation failures, collapsed-profile binding "
                        "candidates, attempts, acceptances, fallback "
                        "acceptances, rejection reasons, profile-diversity "
                        "deltas, update-shape counts, replay plan, "
                        "branch-context gate, coverage floor, diversity target, "
                        "recipe, verifier, and constraint-first promotion "
                        "artifacts."
                    ),
                    "required": True,
                }
            )
    gates.extend(
        parse_experiment_gate(raw_gate)
        for raw_gate in (getattr(args, "experiment_acceptance_gate", None) or [])
    )
    return gates


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


def transformer_training_recipe(
    args: Any,
    tokenizer: Any,
    planned_artifacts: list[Path],
    acceptance_gates: list[dict[str, Any]],
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    generation_config: dict[str, Any],
    replay_plan_path: Path | None = None,
) -> dict[str, Any]:
    direct_enabled = args.direct_answer_steps > 0
    return build_training_recipe(
        version=getattr(args, "experiment_version", TRANSFORMER_RECIPE_VERSION),
        component="transformer-answer-train",
        run_id=args.run.name,
        recipe_id=transformer_training_recipe_id(args),
        purpose=(
            "Train a tiny decoder-only transformer from admitted corpus text and "
            "evaluate reliable-answer behavior under constraint-first gates."
        ),
        model={
            "architecture": TRANSFORMER_ARCHITECTURE,
            "config": dict(model_config),
            "initialization": (
                "declared QuarkLM checkpoint"
                if args.resume_checkpoint is not None
                else "random"
            ),
            "resume_checkpoint": (
                str(args.resume_checkpoint)
                if args.resume_checkpoint is not None
                else None
            ),
            "pretrained_weights": False,
        },
        tokenizer={
            "type": TRANSFORMER_TOKENIZER,
            "source": str(args.train_text),
            "vocab_size": tokenizer.vocab_size,
            "pretrained_tokenizer": False,
        },
        data={
            "train_text": str(args.train_text),
            "valid_text": str(args.valid),
            "corpus_dir": str(args.corpus_dir),
            "eval_sets": [str(path) for path in DEFAULT_ANSWER_EVALS],
            "training_examples": TRAINING_DATA_DESCRIPTION,
        },
        objective={
            "target_loss": {
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "eval_every": args.eval_every,
                "target_loss_weight": args.target_loss_weight,
                "choice_loss_weight": args.choice_loss_weight,
                "choice_negatives": args.choice_negatives,
            },
            "direct_answer": {
                "enabled": direct_enabled,
                "steps": args.direct_answer_steps,
                "mode": args.direct_answer_mode,
                "learning_rate": args.direct_answer_learning_rate,
                "branch_position": args.direct_answer_branch_position,
                "branch_span": args.direct_answer_branch_span,
                "snapshot_mode": args.direct_answer_snapshot_mode,
                "require_branch_context_gate": (
                    args.direct_answer_require_branch_context_gate
                ),
            },
            "generation": dict(generation_config),
        },
        optimizer=dict(optimizer_config),
        replay={
            "status": "planned" if replay_plan_path is not None else "not_applicable",
            "path": str(replay_plan_path) if replay_plan_path is not None else None,
            "profile_aware": (
                direct_enabled
                and is_profile_aware_direct_answer_mode(args.direct_answer_mode)
            ),
        },
        artifacts=planned_artifacts,
        gates=acceptance_gates,
        rerun={
            "entry_point": "quark-lm-transformer answer-train",
            "arguments": {
                "train_text": str(args.train_text),
                "valid": str(args.valid),
                "corpus_dir": str(args.corpus_dir),
                "run": str(args.run),
                "steps": args.steps,
                "context_size": args.context_size,
                "embedding_dim": args.embedding_dim,
                "feedforward_dim": args.feedforward_dim,
                "direct_answer_steps": args.direct_answer_steps,
                "direct_answer_mode": args.direct_answer_mode,
                "seed": args.seed,
            },
        },
        notes=["Recipe uses admitted corpus text, corpus-trained tokenizer, and no external model."],
    )


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
    direct_answer = metrics.get("direct_answer")
    if isinstance(direct_answer, dict):
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
