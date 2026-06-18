"""Artifact path surface for transformer answer-training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    replay_mixture_report: Path
    sweep_plan: Path
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
            replay_mixture_report=run_dir / "replay_mixture_report.json",
            sweep_plan=run_dir / "sweep_plan.json",
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
            self.replay_mixture_report,
            self.sweep_plan,
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
            self.replay_mixture_report,
            self.sweep_plan,
            self.metrics,
            self.metrics_history,
            self.lessons,
            self.experiment_intent,
        ]
        if self.replay_plan is not None:
            paths.append(self.replay_plan)
        return [str(path) for path in paths]
