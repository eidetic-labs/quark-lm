"""Small training-loop utilities for transformer runs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence


class JsonlHistoryWriter:
    """Append deterministic JSONL training snapshots."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record


class ShuffledTrainingCursor:
    """Cycle through a shuffled training pool without leaking ordering policy."""

    def __init__(self, pool: Sequence[Any], rng: random.Random) -> None:
        if not pool:
            raise ValueError("training pool must not be empty")
        self._rng = rng
        self._order = list(pool)
        self._index = 0
        self._rng.shuffle(self._order)

    def next(self) -> Any:
        if self._index == len(self._order):
            self._rng.shuffle(self._order)
            self._index = 0
        item = self._order[self._index]
        self._index += 1
        return item


class LossAccumulator:
    """Track summed training losses and emit averages at snapshot boundaries."""

    def __init__(self) -> None:
        self.loss = 0.0
        self.target_loss = 0.0
        self.choice_loss = 0.0
        self.choice_candidates = 0.0

    def add(
        self,
        loss: float,
        target_loss: float | None = None,
        choice_loss: float | None = None,
        choice_candidates: float | None = None,
    ) -> None:
        self.loss += loss
        self.target_loss += loss if target_loss is None else target_loss
        if choice_loss is not None:
            self.choice_loss += choice_loss
        if choice_candidates is not None:
            self.choice_candidates += choice_candidates

    def average(
        self,
        window: int,
        include_choice: bool = False,
    ) -> dict[str, float | None]:
        if window <= 0:
            raise ValueError("window must be positive")
        return {
            "train_loss": self.loss / window,
            "train_target_loss": self.target_loss / window,
            "train_choice_loss": (
                self.choice_loss / window if include_choice else None
            ),
            "train_choice_candidates": (
                self.choice_candidates / window if include_choice else None
            ),
        }

    def reset(self) -> None:
        self.loss = 0.0
        self.target_loss = 0.0
        self.choice_loss = 0.0
        self.choice_candidates = 0.0
