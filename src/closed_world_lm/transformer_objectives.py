"""Objective-selection primitives for transformer direct-answer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


DirectAnswerTrainer = Callable[..., float]


DIRECT_ANSWER_OBJECTIVE_MODES = [
    "first-error",
    "first-error-unlikelihood",
    "random-char",
    "rollout-unlikelihood",
    "hybrid-unlikelihood",
    "staged-unlikelihood",
    "periodic-rollout-unlikelihood",
    "early-stop-unlikelihood",
    "periodic-early-stop-unlikelihood",
    "repeat-loop-unlikelihood",
    "periodic-repeat-loop-unlikelihood",
    "balanced-repair-unlikelihood",
    "periodic-balanced-repair-unlikelihood",
    "generated-prefix-recovery-unlikelihood",
    "periodic-generated-prefix-recovery-unlikelihood",
    "sequence-repair-unlikelihood",
    "periodic-sequence-repair-unlikelihood",
    "loop-escape-unlikelihood",
    "periodic-loop-escape-unlikelihood",
    "periodic-sequence-loop-escape-unlikelihood",
    "branch-repair-unlikelihood",
    "periodic-branch-repair-unlikelihood",
    "branch-collapse-unlikelihood",
    "periodic-branch-collapse-unlikelihood",
    "branch-batch-contrast-unlikelihood",
    "periodic-branch-batch-contrast-unlikelihood",
    "branch-diversity-unlikelihood",
    "periodic-branch-diversity-unlikelihood",
    "branch-target-softmax-unlikelihood",
    "periodic-branch-target-softmax-unlikelihood",
    "branch-target-margin-unlikelihood",
    "periodic-branch-target-margin-unlikelihood",
    "branch-representation-contrast-unlikelihood",
    "branch-balanced-representation-contrast-unlikelihood",
    "branch-output-binding-unlikelihood",
    "branch-bidirectional-binding-unlikelihood",
    "branch-balanced-bidirectional-binding-unlikelihood",
    "branch-coverage-binding-unlikelihood",
    "branch-balanced-coverage-binding-unlikelihood",
    "branch-target-set-coverage-unlikelihood",
    "branch-balanced-target-set-coverage-unlikelihood",
    "branch-target-diversity-unlikelihood",
    "branch-balanced-target-diversity-unlikelihood",
    "branch-target-replay-coverage-unlikelihood",
    "branch-balanced-target-replay-coverage-unlikelihood",
    "branch-context-replay-coverage-unlikelihood",
    "branch-balanced-context-replay-coverage-unlikelihood",
    "branch-context-coverage-anchor-unlikelihood",
    "branch-balanced-context-coverage-anchor-unlikelihood",
    "branch-context-target-balanced-anchor-unlikelihood",
    "branch-balanced-context-target-balanced-anchor-unlikelihood",
    "branch-context-coverage-deficit-unlikelihood",
    "branch-balanced-context-coverage-deficit-unlikelihood",
    "branch-context-coverage-preserving-deficit-unlikelihood",
    "branch-balanced-context-coverage-preserving-deficit-unlikelihood",
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
    "branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-coverage-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-coverage-prep-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-coverage-recovery-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-branch-diversity-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-branch-diversity-collapsed-profile-binding-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-branch-diversity-collapsed-profile-binding-remaining-profile-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-branch-diversity-collapsed-profile-binding-remaining-profile-owner-paraphrase-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-branch-diversity-collapsed-profile-binding-remaining-profile-owner-paraphrase-memory-consolidation-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-branch-diversity-collapsed-profile-binding-remaining-profile-owner-paraphrase-memory-consolidation-missing-first-token-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-recovery-branch-diversity-collapsed-profile-binding-remaining-profile-owner-paraphrase-memory-consolidation-remaining-collapsed-missing-first-token-frontier-profile-scale-calibrated-sequential-profile-stabilization-unlikelihood",
    "branch-rank-margin-unlikelihood",
    "branch-balanced-rank-margin-unlikelihood",
    "branch-topk-softmax-unlikelihood",
    "branch-balanced-topk-softmax-unlikelihood",
    "periodic-branch-representation-contrast-unlikelihood",
    "branch-span-repair-unlikelihood",
    "periodic-branch-span-repair-unlikelihood",
    "branch-contrast-unlikelihood",
    "periodic-branch-contrast-unlikelihood",
    "branch-span-contrast-unlikelihood",
    "periodic-branch-span-contrast-unlikelihood",
    "hard-branch-contrast-unlikelihood",
    "periodic-hard-branch-contrast-unlikelihood",
    "periodic-branch-repair-contrast-unlikelihood",
    "periodic-branch-span-repair-contrast-unlikelihood",
    "periodic-hard-branch-repair-contrast-unlikelihood",
]


PERIODIC_DIRECT_ANSWER_OBJECTIVE_MODES = [
    mode for mode in DIRECT_ANSWER_OBJECTIVE_MODES if mode.startswith("periodic-")
]


def validate_direct_answer_objective_mode(mode: str) -> str:
    if mode not in DIRECT_ANSWER_OBJECTIVE_MODES:
        known = ", ".join(DIRECT_ANSWER_OBJECTIVE_MODES)
        raise ValueError(f"unknown direct-answer objective mode: {mode}; known modes: {known}")
    return mode


@dataclass(frozen=True)
class DirectAnswerObjective:
    name: str
    trainer: DirectAnswerTrainer
    rule: str

    def train(self, **kwargs: Any) -> float:
        return self.trainer(**kwargs)


class DirectAnswerObjectiveRegistry:
    def __init__(self) -> None:
        self._objectives: dict[str, DirectAnswerObjective] = {}

    def register(
        self,
        name: str,
        trainer: DirectAnswerTrainer,
        rule: str,
    ) -> None:
        if not name:
            raise ValueError("objective name must not be empty")
        if name in self._objectives:
            raise ValueError(f"duplicate direct-answer objective: {name}")
        self._objectives[name] = DirectAnswerObjective(name, trainer, rule)

    def get(self, name: str) -> DirectAnswerObjective:
        try:
            return self._objectives[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._objectives))
            raise ValueError(
                f"unknown direct-answer objective: {name}; known modes: {known}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._objectives)

    def rules(self) -> dict[str, str]:
        return {name: objective.rule for name, objective in self._objectives.items()}


def staged_unlikelihood_objective_name(step: int, total_steps: int) -> str:
    if step <= total_steps // 2:
        return "first-error-unlikelihood"
    return "rollout-unlikelihood"


def interval_choice(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0
