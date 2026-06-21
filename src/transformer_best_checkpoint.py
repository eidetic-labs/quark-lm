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
