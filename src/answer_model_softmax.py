"""Softmax classifier used by the closed-world answer model."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from answer_examples import AnswerExample
from answer_features import feature_names


@dataclass
class AnswerModelConfig:
    labels: list[str]
    features: list[str]
    seed: int = 7


class AnswerSoftmax:
    def __init__(
        self,
        config: AnswerModelConfig,
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
    def init_random(cls, config: AnswerModelConfig) -> "AnswerSoftmax":
        rng = random.Random(config.seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in config.features]
            for _ in config.labels
        ]
        bias = [0.0 for _ in config.labels]
        return cls(config, weights, bias)

    def featurize(self, prompt: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for name in feature_names(prompt):
            index = self.feature_to_index.get(name)
            if index is not None:
                counts[index] = counts.get(index, 0.0) + 1.0
        return counts

    def probabilities(self, prompt: str) -> list[float]:
        features = self.featurize(prompt)
        logits = self._logits(features)
        return softmax(logits)

    def predict(self, prompt: str) -> str:
        probs = self.probabilities(prompt)
        index = max(range(len(probs)), key=lambda item: probs[item])
        return self.config.labels[index]

    def loss(self, prompt: str, target: str) -> float:
        target_index = self.label_to_index[target]
        probs = self.probabilities(prompt)
        return -math.log(max(probs[target_index], 1e-12))

    def train_step(self, example: AnswerExample, learning_rate: float) -> float:
        target_index = self.label_to_index[example.target]
        features = self.featurize(example.prompt)
        logits = self._logits(features)
        probs = softmax(logits)
        loss = -math.log(max(probs[target_index], 1e-12))
        probs[target_index] -= 1.0

        for label_index, grad in enumerate(probs):
            self.bias[label_index] -= learning_rate * grad
            for feature_index, value in features.items():
                self.weights[label_index][feature_index] -= learning_rate * grad * value
        return loss

    def _logits(self, features: dict[int, float]) -> list[float]:
        logits = self.bias[:]
        for label_index, row in enumerate(self.weights):
            total = logits[label_index]
            for feature_index, value in features.items():
                total += row[feature_index] * value
            logits[label_index] = total
        return logits

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerSoftmax":
        return cls(
            AnswerModelConfig(**payload["config"]),
            payload["weights"],
            payload["bias"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "AnswerSoftmax":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(item - max_logit) for item in logits]
    total = sum(exps)
    return [item / total for item in exps]
