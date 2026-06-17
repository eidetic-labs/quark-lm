"""Checkpointable transformer-guided answer generator."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_answer_generator_config import TransformerAnswerGeneratorConfig
from transformer_answer_generator_constants import GENERATOR_EOS
from transformer_answer_generator_features import (
    transformer_answer_generator_feature_names,
)
from transformer_math import softmax_floats


class TransformerGuidedAnswerGenerator:
    """Prompt-conditioned character generator with transformer-derived features."""

    def __init__(
        self,
        config: TransformerAnswerGeneratorConfig,
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
    def init_random(
        cls,
        config: TransformerAnswerGeneratorConfig,
    ) -> "TransformerGuidedAnswerGenerator":
        rng = random.Random(config.seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in config.features]
            for _ in config.labels
        ]
        return cls(config, weights, [0.0 for _ in config.labels])

    def featurize(
        self,
        model: Any,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> dict[int, float]:
        counts: dict[int, float] = {}
        for name in transformer_answer_generator_feature_names(
            model,
            tokenizer,
            prompt,
            prefix,
            self.config.transformer_top_k,
        ):
            index = self.feature_to_index.get(name)
            if index is None:
                continue
            counts[index] = counts.get(index, 0.0) + 1.0
        return counts

    def probabilities(
        self,
        model: Any,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> list[float]:
        return softmax_floats(self._logits(self.featurize(model, tokenizer, prompt, prefix)))

    def predict_next(
        self,
        model: Any,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> str:
        probs = self.probabilities(model, tokenizer, prompt, prefix)
        index = max(range(len(probs)), key=lambda item: probs[item])
        return self.config.labels[index]

    def generate(
        self,
        model: Any,
        tokenizer: CharTokenizer,
        prompt: str,
    ) -> str:
        prefix = ""
        for _ in range(self.config.max_answer_chars):
            label = self.predict_next(model, tokenizer, prompt, prefix)
            if label == GENERATOR_EOS:
                break
            prefix += label
        return prefix

    def sequence_loss(
        self,
        model: Any,
        tokenizer: CharTokenizer,
        prompt: str,
        target: str,
    ) -> float:
        prefix = ""
        total = 0.0
        labels = [*target, GENERATOR_EOS]
        for label in labels:
            probs = self.probabilities(model, tokenizer, prompt, prefix)
            total += -math.log(max(probs[self.label_to_index[label]], 1e-12))
            if label != GENERATOR_EOS:
                prefix += label
        return total / len(labels)

    def train_example(
        self,
        model: Any,
        tokenizer: CharTokenizer,
        example: AnswerExample,
        learning_rate: float,
    ) -> float:
        prefix = ""
        total = 0.0
        labels = [*example.target, GENERATOR_EOS]
        for label in labels:
            target_index = self.label_to_index[label]
            features = self.featurize(model, tokenizer, example.prompt, prefix)
            probs = softmax_floats(self._logits(features))
            total += -math.log(max(probs[target_index], 1e-12))
            probs[target_index] -= 1.0
            self._apply_gradients(probs, features, learning_rate)
            if label != GENERATOR_EOS:
                prefix += label
        return total / len(labels)

    def _apply_gradients(
        self,
        probs: list[float],
        features: dict[int, float],
        learning_rate: float,
    ) -> None:
        for label_index, grad in enumerate(probs):
            clipped_grad = max(min(grad, 5.0), -5.0)
            self.bias[label_index] -= learning_rate * clipped_grad
            for feature_index, value in features.items():
                self.weights[label_index][feature_index] -= (
                    learning_rate * clipped_grad * value
                )

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
            "architecture": "transformer-guided-answer-generator",
            "config": asdict(self.config),
            "weights": self.weights,
            "bias": self.bias,
            "pretrained_weights": False,
            "external_embeddings": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformerGuidedAnswerGenerator":
        return cls(
            TransformerAnswerGeneratorConfig(**payload["config"]),
            payload["weights"],
            payload["bias"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "TransformerGuidedAnswerGenerator":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
