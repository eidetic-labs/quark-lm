"""Closed-world answer candidate selection."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from answer_model import AnswerExample, feature_names
from transformer_math import softmax_floats


@dataclass
class AnswerSelectorConfig:
    labels: list[str]
    features: list[str]
    seed: int = 17


class AnswerCandidateSelector:
    """Small closed-world candidate selector paired with transformer evidence."""

    def __init__(
        self,
        config: AnswerSelectorConfig,
        weights: list[list[float]],
        bias: list[float],
    ) -> None:
        self.config = config
        self.weights = weights
        self.bias = bias
        self.label_to_index = {label: index for index, label in enumerate(config.labels)}
        self.feature_to_index = {
            feature: index for index, feature in enumerate(config.features)
        }

    @classmethod
    def init_random(cls, config: AnswerSelectorConfig) -> "AnswerCandidateSelector":
        rng = random.Random(config.seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in config.features]
            for _ in config.labels
        ]
        return cls(config, weights, [0.0 for _ in config.labels])

    def featurize(self, prompt: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for name in feature_names(prompt):
            index = self.feature_to_index.get(name)
            if index is None:
                continue
            counts[index] = counts.get(index, 0.0) + 1.0
        return counts

    def score(self, prompt: str, candidate: str) -> float:
        label_index = self.label_to_index.get(candidate)
        if label_index is None:
            return -math.inf
        return self._logit(label_index, self.featurize(prompt))

    def predict(self, prompt: str, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("candidate selector requires at least one candidate")
        return max(candidates, key=lambda candidate: self.score(prompt, candidate))

    def loss(
        self,
        prompt: str,
        target: str,
        candidates: list[str] | None = None,
    ) -> float:
        candidate_labels = self._candidate_labels(target, candidates)
        features = self.featurize(prompt)
        logits = [
            self._logit(self.label_to_index[label], features)
            for label in candidate_labels
        ]
        probs = softmax_floats(logits)
        target_offset = candidate_labels.index(target)
        return -math.log(max(probs[target_offset], 1e-12))

    def train_step(
        self,
        example: AnswerExample,
        learning_rate: float,
        candidates: list[str] | None = None,
    ) -> float:
        candidate_labels = self._candidate_labels(example.target, candidates)
        features = self.featurize(example.prompt)
        label_indices = [self.label_to_index[label] for label in candidate_labels]
        logits = [self._logit(label_index, features) for label_index in label_indices]
        probs = softmax_floats(logits)
        target_offset = candidate_labels.index(example.target)
        loss = -math.log(max(probs[target_offset], 1e-12))
        probs[target_offset] -= 1.0
        for label_index, grad in zip(label_indices, probs, strict=True):
            clipped_grad = max(min(grad, 5.0), -5.0)
            self.bias[label_index] -= learning_rate * clipped_grad
            for feature_index, value in features.items():
                self.weights[label_index][feature_index] -= (
                    learning_rate * clipped_grad * value
                )
        return loss

    def _candidate_labels(self, target: str, candidates: list[str] | None) -> list[str]:
        labels = self.config.labels if candidates is None else candidates
        unique_labels = [
            label for label in dict.fromkeys(labels) if label in self.label_to_index
        ]
        if target not in unique_labels:
            if target not in self.label_to_index:
                raise ValueError(f"target {target!r} is outside selector labels")
            unique_labels = [target, *unique_labels]
        return unique_labels

    def _logit(self, label_index: int, features: dict[int, float]) -> float:
        total = self.bias[label_index]
        row = self.weights[label_index]
        for feature_index, value in features.items():
            total += row[feature_index] * value
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "closed-world-answer-candidate-selector",
            "config": asdict(self.config),
            "weights": self.weights,
            "bias": self.bias,
            "pretrained_weights": False,
            "external_embeddings": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerCandidateSelector":
        return cls(
            AnswerSelectorConfig(**payload["config"]),
            payload["weights"],
            payload["bias"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "AnswerCandidateSelector":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def build_answer_selector(
    examples: list[AnswerExample],
    seed: int,
) -> AnswerCandidateSelector:
    labels = sorted({example.target for example in examples})
    features: set[str] = set()
    for example in examples:
        features.update(feature_names(example.prompt))
    config = AnswerSelectorConfig(labels=labels, features=sorted(features), seed=seed)
    return AnswerCandidateSelector.init_random(config)
