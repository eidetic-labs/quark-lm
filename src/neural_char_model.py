"""Dependency-free neural character language model.

This is deliberately small and plain: embeddings, one tanh hidden layer, and a
softmax next-character head. It gives the experiment real random weights without
smuggling in a pretrained ML stack.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neural_char_metrics import average_nll, continuation_nll
from neural_char_ops import context_before, make_context, sample_from_probs, softmax
from tokenizer import CharTokenizer


__all__ = [
    "CharMLP",
    "ModelConfig",
    "average_nll",
    "context_before",
    "continuation_nll",
    "make_context",
]


@dataclass
class ModelConfig:
    vocab_size: int
    context_size: int = 64
    embedding_dim: int = 12
    hidden_dim: int = 48
    seed: int = 7


class CharMLP:
    def __init__(
        self,
        config: ModelConfig,
        embeddings: list[list[float]],
        w1: list[list[float]],
        b1: list[float],
        w2: list[list[float]],
        b2: list[float],
    ) -> None:
        self.config = config
        self.embeddings = embeddings
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    @classmethod
    def init_random(cls, config: ModelConfig) -> "CharMLP":
        rng = random.Random(config.seed)

        def rand(scale: float) -> float:
            return rng.uniform(-scale, scale)

        input_dim = config.context_size * config.embedding_dim
        embeddings = [
            [rand(0.05) for _ in range(config.embedding_dim)]
            for _ in range(config.vocab_size)
        ]
        w1 = [[rand(0.05) for _ in range(config.hidden_dim)] for _ in range(input_dim)]
        b1 = [0.0 for _ in range(config.hidden_dim)]
        w2 = [[rand(0.05) for _ in range(config.vocab_size)] for _ in range(config.hidden_dim)]
        b2 = [0.0 for _ in range(config.vocab_size)]
        return cls(config, embeddings, w1, b1, w2, b2)

    def _flatten_context(self, context: list[int]) -> list[float]:
        if len(context) != self.config.context_size:
            raise ValueError(
                f"context must have {self.config.context_size} ids, got {len(context)}"
            )
        flat: list[float] = []
        for token_id in context:
            flat.extend(self.embeddings[token_id])
        return flat

    def _forward(self, context: list[int]) -> tuple[list[float], list[float], list[float]]:
        x = self._flatten_context(context)
        h: list[float] = []
        for hidden_index in range(self.config.hidden_dim):
            total = self.b1[hidden_index]
            for input_index, value in enumerate(x):
                total += value * self.w1[input_index][hidden_index]
            h.append(math.tanh(total))

        logits: list[float] = []
        for vocab_index in range(self.config.vocab_size):
            total = self.b2[vocab_index]
            for hidden_index, value in enumerate(h):
                total += value * self.w2[hidden_index][vocab_index]
            logits.append(total)
        return x, h, softmax(logits)

    def predict(self, context: list[int]) -> list[float]:
        return self._forward(context)[2]

    def nll(self, context: list[int], target: int) -> float:
        probs = self.predict(context)
        return -math.log(max(probs[target], 1e-12))

    def train_step(self, context: list[int], target: int, learning_rate: float) -> float:
        x, h, probs = self._forward(context)
        loss = -math.log(max(probs[target], 1e-12))

        dlogits = probs[:]
        dlogits[target] -= 1.0

        old_w2 = [row[:] for row in self.w2]

        for hidden_index in range(self.config.hidden_dim):
            hidden_value = h[hidden_index]
            for vocab_index in range(self.config.vocab_size):
                grad = hidden_value * dlogits[vocab_index]
                self.w2[hidden_index][vocab_index] -= learning_rate * grad
        for vocab_index in range(self.config.vocab_size):
            self.b2[vocab_index] -= learning_rate * dlogits[vocab_index]

        dh: list[float] = []
        for hidden_index in range(self.config.hidden_dim):
            total = 0.0
            for vocab_index in range(self.config.vocab_size):
                total += old_w2[hidden_index][vocab_index] * dlogits[vocab_index]
            dh.append(total)

        dpre = [dh[index] * (1.0 - h[index] * h[index]) for index in range(self.config.hidden_dim)]

        old_w1 = [row[:] for row in self.w1]
        for input_index, value in enumerate(x):
            for hidden_index in range(self.config.hidden_dim):
                grad = value * dpre[hidden_index]
                self.w1[input_index][hidden_index] -= learning_rate * grad
        for hidden_index in range(self.config.hidden_dim):
            self.b1[hidden_index] -= learning_rate * dpre[hidden_index]

        dx = [0.0 for _ in x]
        for input_index in range(len(x)):
            total = 0.0
            for hidden_index in range(self.config.hidden_dim):
                total += old_w1[input_index][hidden_index] * dpre[hidden_index]
            dx[input_index] = total

        for position, token_id in enumerate(context):
            start = position * self.config.embedding_dim
            for emb_index in range(self.config.embedding_dim):
                grad = dx[start + emb_index]
                self.embeddings[token_id][emb_index] -= learning_rate * grad

        return loss

    def generate(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_new_chars: int,
        temperature: float = 0.0,
    ) -> str:
        ids = tokenizer.encode(prompt)
        generated: list[int] = []
        rng = random.Random(self.config.seed + len(prompt))
        for _ in range(max_new_chars):
            context = make_context(ids, self.config.context_size, tokenizer.pad_id)
            probs = self.predict(context)
            if temperature <= 0:
                next_id = max(range(len(probs)), key=lambda index: probs[index])
            else:
                next_id = sample_from_probs(probs, temperature, rng)
            ids.append(next_id)
            generated.append(next_id)
        return tokenizer.decode(generated)

    def to_dict(self, tokenizer: CharTokenizer | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": asdict(self.config),
            "weights": {
                "embeddings": self.embeddings,
                "w1": self.w1,
                "b1": self.b1,
                "w2": self.w2,
                "b2": self.b2,
            },
        }
        if tokenizer is not None:
            payload["tokenizer"] = tokenizer.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> tuple["CharMLP", CharTokenizer | None]:
        config = ModelConfig(**payload["config"])
        weights = payload["weights"]
        model = cls(
            config=config,
            embeddings=weights["embeddings"],
            w1=weights["w1"],
            b1=weights["b1"],
            w2=weights["w2"],
            b2=weights["b2"],
        )
        tokenizer = None
        if "tokenizer" in payload:
            tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
        return model, tokenizer

    def save(self, path: Path, tokenizer: CharTokenizer | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(tokenizer), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> tuple["CharMLP", CharTokenizer | None]:
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
