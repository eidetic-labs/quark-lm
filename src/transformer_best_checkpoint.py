"""Eval-driven best-checkpoint selection for the torch training loop.

Training to the last step can overshoot the best-generalizing weights. This
tracker watches a validation metric during training and retains a snapshot of the
parameters at the best (lowest) value, so the loop can restore the best checkpoint
instead of the last. It is engaged only when a validation slice is supplied, so
the parity-validated single-example training path is unchanged.
"""

from __future__ import annotations

from typing import Any


class BestCheckpointTracker:
    """Retain a parameter snapshot at the lowest validation loss seen."""

    def __init__(self) -> None:
        self.best_loss: float | None = None
        self.best_step: int | None = None
        self._snapshot: list[Any] | None = None

    def consider(self, step: int, validation_loss: float, params: list[Any]) -> None:
        """Snapshot the current params iff this is the best validation loss so far.

        Strict ``<`` keeps the EARLIEST step at a tied minimum (deterministic).
        """

        if self.best_loss is None or validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_step = step
            self._snapshot = [parameter.detach().clone() for parameter in params]

    def restore(self, params: list[Any]) -> bool:
        """Copy the best snapshot back onto params in place; False if none taken."""

        if self._snapshot is None:
            return False
        for parameter, saved in zip(params, self._snapshot):
            parameter.data.copy_(saved)
        return True


class CombinedBestCheckpointTracker:
    """Retain a parameter snapshot at the best gated does-both score.

    Selects the checkpoint that BOTH abstains and generates concrete answers.
    The score is a gated harmonic mean of abstention_f1 and concrete_gen: 0.0
    unless BOTH clear their floors, else 2*f1*gen/(f1+gen). The harmonic mean
    refuses the seesaw -- it collapses toward whichever axis is weaker, so an
    over-abstainer (gen=0) scores 0 and cannot masquerade as success the way a
    plain mean (which would reward abstain-everywhere) would. HIGHER is better;
    strict ``>`` keeps the EARLIEST step at a tied maximum (deterministic).
    """

    def __init__(self, f1_floor: float, gen_floor: float) -> None:
        self.f1_floor = f1_floor
        self.gen_floor = gen_floor
        self.best_score: float | None = None
        self.best_step: int | None = None
        self.best_f1: float | None = None
        self.best_gen: float | None = None
        self._snapshot: list[Any] | None = None

    def _score(self, abstention_f1: float, concrete_gen: float) -> float:
        if abstention_f1 < self.f1_floor or concrete_gen < self.gen_floor:
            return 0.0
        if abstention_f1 + concrete_gen <= 0.0:
            return 0.0
        return 2.0 * abstention_f1 * concrete_gen / (abstention_f1 + concrete_gen)

    def consider(
        self,
        step: int,
        *,
        abstention_f1: float,
        concrete_gen: float,
        params: list[Any],
    ) -> float:
        """Snapshot params iff this is the best gated does-both score so far.

        Returns the computed score. A score of 0.0 (either floor missed) is never
        snapshotted, so a run where no step clears both floors leaves restore() a
        no-op -- the loop must then fail closed rather than keep last-step weights.
        """

        score = self._score(abstention_f1, concrete_gen)
        if score > 0.0 and (self.best_score is None or score > self.best_score):
            self.best_score = score
            self.best_step = step
            self.best_f1 = abstention_f1
            self.best_gen = concrete_gen
            self._snapshot = [parameter.detach().clone() for parameter in params]
        return score

    def restore(self, params: list[Any]) -> bool:
        """Copy the best snapshot back onto params in place; False if none taken."""

        if self._snapshot is None:
            return False
        for parameter, saved in zip(params, self._snapshot):
            parameter.data.copy_(saved)
        return True
