"""Dependency-free tiny decoder-only transformer language model.

The implementation is intentionally small and auditable. It uses learned token
and position embeddings, one causal self-attention block, a feed-forward block,
and a next-character language-model head. All weights start from random values;
the tokenizer is trained from admitted corpus text.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .autograd import Scalar, zero_grad
from .answer_model import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVALS as DEFAULT_ANSWER_EVALS,
    DEFAULT_TRAIN_TEXT,
    AnswerExample,
    answer_training_pool,
    feature_names,
    load_training_examples,
    semantic_feature_names,
    write_lessons,
)
from .curriculum import DEFAULT_OUTPUT_DIR, build_curriculum, write_curriculum
from .neural_char_model import context_before, continuation_nll, make_context
from .probes import read_jsonl, score_records, summarize
from .tokenizer import CharTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "transformer-latest"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "transformer.json"
DEFAULT_PROBES = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "paraphrases.jsonl",
]
ANSWER_TERMINATOR = "\n"


@dataclass
class TransformerConfig:
    vocab_size: int
    context_size: int = 16
    embedding_dim: int = 8
    feedforward_dim: int = 16
    seed: int = 17
    use_layer_norm: bool = False
    layer_norm_epsilon: float = 1e-5


class TinyTransformerLM:
    def __init__(self, config: TransformerConfig, weights: dict[str, Any]) -> None:
        self.config = config
        self.token_embeddings = matrix_to_scalars(weights["token_embeddings"])
        self.position_embeddings = matrix_to_scalars(weights["position_embeddings"])
        self.wq = matrix_to_scalars(weights["wq"])
        self.bq = vector_to_scalars(weights["bq"])
        self.wk = matrix_to_scalars(weights["wk"])
        self.bk = vector_to_scalars(weights["bk"])
        self.wv = matrix_to_scalars(weights["wv"])
        self.bv = vector_to_scalars(weights["bv"])
        self.wo = matrix_to_scalars(weights["wo"])
        self.bo = vector_to_scalars(weights["bo"])
        self.w1 = matrix_to_scalars(weights["w1"])
        self.b1 = vector_to_scalars(weights["b1"])
        self.w2 = matrix_to_scalars(weights["w2"])
        self.b2 = vector_to_scalars(weights["b2"])
        self.wout = matrix_to_scalars(weights["wout"])
        self.bout = vector_to_scalars(weights["bout"])
        dim = config.embedding_dim
        self.ln1_gain = vector_to_scalars(weights.get("ln1_gain", [1.0 for _ in range(dim)]))
        self.ln1_bias = vector_to_scalars(weights.get("ln1_bias", [0.0 for _ in range(dim)]))
        self.ln2_gain = vector_to_scalars(weights.get("ln2_gain", [1.0 for _ in range(dim)]))
        self.ln2_bias = vector_to_scalars(weights.get("ln2_bias", [0.0 for _ in range(dim)]))

    @classmethod
    def init_random(cls, config: TransformerConfig) -> "TinyTransformerLM":
        rng = random.Random(config.seed)

        def rand(scale: float) -> float:
            return rng.uniform(-scale, scale)

        dim = config.embedding_dim
        ff_dim = config.feedforward_dim
        scale = 1.0 / math.sqrt(dim)
        weights = {
            "token_embeddings": [
                [rand(0.08) for _ in range(dim)]
                for _ in range(config.vocab_size)
            ],
            "position_embeddings": [
                [rand(0.08) for _ in range(dim)]
                for _ in range(config.context_size)
            ],
            "wq": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
            "bq": [0.0 for _ in range(dim)],
            "wk": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
            "bk": [0.0 for _ in range(dim)],
            "wv": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
            "bv": [0.0 for _ in range(dim)],
            "wo": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
            "bo": [0.0 for _ in range(dim)],
            "w1": [[rand(scale) for _ in range(ff_dim)] for _ in range(dim)],
            "b1": [0.0 for _ in range(ff_dim)],
            "w2": [[rand(1.0 / math.sqrt(ff_dim)) for _ in range(dim)] for _ in range(ff_dim)],
            "b2": [0.0 for _ in range(dim)],
            "wout": [[rand(scale) for _ in range(config.vocab_size)] for _ in range(dim)],
            "bout": [0.0 for _ in range(config.vocab_size)],
            "ln1_gain": [1.0 for _ in range(dim)],
            "ln1_bias": [0.0 for _ in range(dim)],
            "ln2_gain": [1.0 for _ in range(dim)],
            "ln2_bias": [0.0 for _ in range(dim)],
        }
        return cls(config, weights)

    def parameters(self) -> list[Scalar]:
        params: list[Scalar] = []
        for item in [
            self.token_embeddings,
            self.position_embeddings,
            self.wq,
            self.bq,
            self.wk,
            self.bk,
            self.wv,
            self.bv,
            self.wo,
            self.bo,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.wout,
            self.bout,
        ]:
            params.extend(flatten_scalars(item))
        if self.config.use_layer_norm:
            for item in [self.ln1_gain, self.ln1_bias, self.ln2_gain, self.ln2_bias]:
                params.extend(flatten_scalars(item))
        return params

    def _forward_scalars(self, context: list[int]) -> list[Scalar]:
        if len(context) != self.config.context_size:
            raise ValueError(
                f"context must have {self.config.context_size} ids, got {len(context)}"
            )
        x = [
            [
                self.token_embeddings[token_id][dim] + self.position_embeddings[position][dim]
                for dim in range(self.config.embedding_dim)
            ]
            for position, token_id in enumerate(context)
        ]
        q = [linear_scalars(row, self.wq, self.bq) for row in x]
        k = [linear_scalars(row, self.wk, self.bk) for row in x]
        v = [linear_scalars(row, self.wv, self.bv) for row in x]
        scale = 1.0 / math.sqrt(self.config.embedding_dim)
        last_position = self.config.context_size - 1
        scores = [dot_scalars(q[last_position], k[past]) * scale for past in range(self.config.context_size)]
        weights = softmax_scalars(scores)
        attended = []
        for dim in range(self.config.embedding_dim):
            total = Scalar(0.0)
            for past, weight in enumerate(weights):
                total = total + weight * v[past][dim]
            attended.append(total)
        projected = linear_scalars(attended, self.wo, self.bo)
        hidden = [
            x[last_position][dim] + projected[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            hidden = layer_norm_scalars(
                hidden,
                self.ln1_gain,
                self.ln1_bias,
                self.config.layer_norm_epsilon,
            )
        ff_hidden = [value.tanh() for value in linear_scalars(hidden, self.w1, self.b1)]
        ff_out = linear_scalars(ff_hidden, self.w2, self.b2)
        block_out = [
            hidden[dim] + ff_out[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            block_out = layer_norm_scalars(
                block_out,
                self.ln2_gain,
                self.ln2_bias,
                self.config.layer_norm_epsilon,
            )
        return linear_scalars(block_out, self.wout, self.bout)

    def _forward_floats(self, context: list[int]) -> list[float]:
        if len(context) != self.config.context_size:
            raise ValueError(
                f"context must have {self.config.context_size} ids, got {len(context)}"
            )
        token_embeddings = matrix_to_floats(self.token_embeddings)
        position_embeddings = matrix_to_floats(self.position_embeddings)
        wq = matrix_to_floats(self.wq)
        bq = vector_to_floats(self.bq)
        wk = matrix_to_floats(self.wk)
        bk = vector_to_floats(self.bk)
        wv = matrix_to_floats(self.wv)
        bv = vector_to_floats(self.bv)
        wo = matrix_to_floats(self.wo)
        bo = vector_to_floats(self.bo)
        w1 = matrix_to_floats(self.w1)
        b1 = vector_to_floats(self.b1)
        w2 = matrix_to_floats(self.w2)
        b2 = vector_to_floats(self.b2)
        wout = matrix_to_floats(self.wout)
        bout = vector_to_floats(self.bout)
        ln1_gain = vector_to_floats(self.ln1_gain)
        ln1_bias = vector_to_floats(self.ln1_bias)
        ln2_gain = vector_to_floats(self.ln2_gain)
        ln2_bias = vector_to_floats(self.ln2_bias)
        x = [
            [
                token_embeddings[token_id][dim] + position_embeddings[position][dim]
                for dim in range(self.config.embedding_dim)
            ]
            for position, token_id in enumerate(context)
        ]
        q = [linear_floats(row, wq, bq) for row in x]
        k = [linear_floats(row, wk, bk) for row in x]
        v = [linear_floats(row, wv, bv) for row in x]
        scale = 1.0 / math.sqrt(self.config.embedding_dim)
        last_position = self.config.context_size - 1
        scores = [dot_floats(q[last_position], k[past]) * scale for past in range(self.config.context_size)]
        weights = softmax_floats(scores)
        attended = [
            sum(weight * v[past][dim] for past, weight in enumerate(weights))
            for dim in range(self.config.embedding_dim)
        ]
        projected = linear_floats(attended, wo, bo)
        hidden = [
            x[last_position][dim] + projected[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            hidden = layer_norm_floats(
                hidden,
                ln1_gain,
                ln1_bias,
                self.config.layer_norm_epsilon,
            )
        ff_hidden = [math.tanh(value) for value in linear_floats(hidden, w1, b1)]
        ff_out = linear_floats(ff_hidden, w2, b2)
        block_out = [
            hidden[dim] + ff_out[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            block_out = layer_norm_floats(
                block_out,
                ln2_gain,
                ln2_bias,
                self.config.layer_norm_epsilon,
            )
        return linear_floats(block_out, wout, bout)

    def predict(self, context: list[int]) -> list[float]:
        return softmax_floats(self._forward_floats(context))

    def nll(self, context: list[int], target: int) -> float:
        probs = self.predict(context)
        return -math.log(max(probs[target], 1e-12))

    def train_step(
        self,
        context: list[int],
        target: int,
        learning_rate: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        loss = cross_entropy_scalars(self._forward_scalars(context), target)
        loss.backward()
        for parameter in params:
            clipped_grad = max(min(parameter.grad, 5.0), -5.0)
            parameter.data -= learning_rate * clipped_grad
        return loss.data

    def train_step_with_unlikelihood(
        self,
        context: list[int],
        target: int,
        negative: int,
        learning_rate: float,
        negative_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if negative != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[negative] + 1e-12).log()) * negative_weight
        loss.backward()
        for parameter in params:
            clipped_grad = max(min(parameter.grad, 5.0), -5.0)
            parameter.data -= learning_rate * clipped_grad
        return loss.data

    def train_step_with_unlikelihood_and_positive(
        self,
        context: list[int],
        target: int,
        negative: int,
        positive_context: list[int],
        positive_target: int,
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if negative != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[negative] + 1e-12).log()) * negative_weight
        if positive_weight > 0.0:
            positive_probs = softmax_scalars(self._forward_scalars(positive_context))
            loss = loss + (-positive_probs[positive_target].log()) * positive_weight
        loss.backward()
        for parameter in params:
            clipped_grad = max(min(parameter.grad, 5.0), -5.0)
            parameter.data -= learning_rate * clipped_grad
        return loss.data

    def train_step_with_branch_contrast(
        self,
        context: list[int],
        target: int,
        contrast_context: list[int],
        contrast_target: int,
        learning_rate: float,
        negative_weight: float,
        contrast_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if contrast_target != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[contrast_target] + 1e-12).log()) * negative_weight
        if contrast_weight > 0.0:
            contrast_probs = softmax_scalars(self._forward_scalars(contrast_context))
            loss = loss + (-contrast_probs[contrast_target].log()) * contrast_weight
            if target != contrast_target and negative_weight > 0.0:
                loss = loss + (
                    -(Scalar(1.0) - contrast_probs[target] + 1e-12).log()
                ) * negative_weight * contrast_weight
        loss.backward()
        for parameter in params:
            clipped_grad = max(min(parameter.grad, 5.0), -5.0)
            parameter.data -= learning_rate * clipped_grad
        return loss.data

    def generate(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_new_chars: int,
        temperature: float = 0.0,
        stop_at: str | None = None,
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
            if stop_at is not None and tokenizer.itos[next_id] == stop_at:
                break
            generated.append(next_id)
        return tokenizer.decode(generated)

    def to_dict(self, tokenizer: CharTokenizer | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "architecture": "tiny-decoder-only-transformer",
            "config": asdict(self.config),
            "weights": {
                "token_embeddings": matrix_to_floats(self.token_embeddings),
                "position_embeddings": matrix_to_floats(self.position_embeddings),
                "wq": matrix_to_floats(self.wq),
                "bq": vector_to_floats(self.bq),
                "wk": matrix_to_floats(self.wk),
                "bk": vector_to_floats(self.bk),
                "wv": matrix_to_floats(self.wv),
                "bv": vector_to_floats(self.bv),
                "wo": matrix_to_floats(self.wo),
                "bo": vector_to_floats(self.bo),
                "w1": matrix_to_floats(self.w1),
                "b1": vector_to_floats(self.b1),
                "w2": matrix_to_floats(self.w2),
                "b2": vector_to_floats(self.b2),
                "wout": matrix_to_floats(self.wout),
                "bout": vector_to_floats(self.bout),
                "ln1_gain": vector_to_floats(self.ln1_gain),
                "ln1_bias": vector_to_floats(self.ln1_bias),
                "ln2_gain": vector_to_floats(self.ln2_gain),
                "ln2_bias": vector_to_floats(self.ln2_bias),
            },
        }
        if tokenizer is not None:
            payload["tokenizer"] = tokenizer.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> tuple["TinyTransformerLM", CharTokenizer | None]:
        config = TransformerConfig(**payload["config"])
        model = cls(config, payload["weights"])
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
    def load(cls, path: Path) -> tuple["TinyTransformerLM", CharTokenizer | None]:
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


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
        self.feature_to_index = {feature: index for index, feature in enumerate(config.features)}

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

    def loss(self, prompt: str, target: str, candidates: list[str] | None = None) -> float:
        candidate_labels = self._candidate_labels(target, candidates)
        features = self.featurize(prompt)
        logits = [self._logit(self.label_to_index[label], features) for label in candidate_labels]
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
                self.weights[label_index][feature_index] -= learning_rate * clipped_grad * value
        return loss

    def _candidate_labels(self, target: str, candidates: list[str] | None) -> list[str]:
        labels = self.config.labels if candidates is None else candidates
        unique_labels = [label for label in dict.fromkeys(labels) if label in self.label_to_index]
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


GENERATOR_EOS = "<eos>"
GENERATOR_BOS = "<bos>"


@dataclass
class TransformerAnswerGeneratorConfig:
    labels: list[str]
    features: list[str]
    seed: int = 17
    max_answer_chars: int = 64
    transformer_top_k: int = 3


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
        self.feature_to_index = {feature: index for index, feature in enumerate(config.features)}

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
        model: TinyTransformerLM,
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
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> list[float]:
        return softmax_floats(self._logits(self.featurize(model, tokenizer, prompt, prefix)))

    def predict_next(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> str:
        probs = self.probabilities(model, tokenizer, prompt, prefix)
        index = max(range(len(probs)), key=lambda item: probs[item])
        return self.config.labels[index]

    def generate(
        self,
        model: TinyTransformerLM,
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
        model: TinyTransformerLM,
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
        model: TinyTransformerLM,
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
            for label_index, grad in enumerate(probs):
                clipped_grad = max(min(grad, 5.0), -5.0)
                self.bias[label_index] -= learning_rate * clipped_grad
                for feature_index, value in features.items():
                    self.weights[label_index][feature_index] -= (
                        learning_rate * clipped_grad * value
                    )
            if label != GENERATOR_EOS:
                prefix += label
        return total / len(labels)

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


def transformer_answer_generator_feature_names(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    prefix: str,
    transformer_top_k: int,
) -> list[str]:
    names = feature_names(prompt)
    previous = prefix[-1] if prefix else GENERATOR_BOS
    previous_two = prefix[-2:] if len(prefix) >= 2 else GENERATOR_BOS
    names.extend(
        [
            f"pos:{len(prefix)}",
            f"prev:{previous}",
            f"prev2:{previous_two}",
            f"prefix:{prefix}",
        ]
    )
    context_ids = tokenizer.encode(prompt + prefix)
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    probs = model.predict(context)
    top_count = max(0, min(transformer_top_k, len(probs)))
    top_ids = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)[
        :top_count
    ]
    for rank, token_id in enumerate(top_ids):
        token = tokenizer.itos[token_id]
        names.append(f"transformer_top:{rank}:{token!r}")
        if rank == 0:
            names.append(f"transformer_argmax:{token!r}")
    return names


def build_transformer_answer_generator(
    examples: list[AnswerExample],
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    seed: int,
    max_answer_chars: int,
    transformer_top_k: int,
) -> TransformerGuidedAnswerGenerator:
    labels = sorted({char for example in examples for char in example.target} | {GENERATOR_EOS})
    features: set[str] = set()
    for example in examples:
        prefix = ""
        for label in [*example.target, GENERATOR_EOS]:
            features.update(
                transformer_answer_generator_feature_names(
                    model,
                    tokenizer,
                    example.prompt,
                    prefix,
                    transformer_top_k,
                )
            )
            if label != GENERATOR_EOS:
                prefix += label
    config = TransformerAnswerGeneratorConfig(
        labels=labels,
        features=sorted(features),
        seed=seed,
        max_answer_chars=max_answer_chars,
        transformer_top_k=transformer_top_k,
    )
    return TransformerGuidedAnswerGenerator.init_random(config)


def transformer_answer_generator_training_pool(
    examples: list[AnswerExample],
) -> list[AnswerExample]:
    pool: list[AnswerExample] = []
    for example in examples:
        repeats = 1 + len(example.target) // 32
        if example.target != " unknown.":
            repeats += 1
        if (
            example.source.startswith("qa:")
            or example.source.startswith("fact:")
            or example.source.startswith("bridge:")
        ):
            repeats += 2
        if example.source.endswith(":place") or example.source.endswith(":color"):
            repeats += 4
        if example.source.endswith(":owner") or example.source.endswith(":training_data"):
            repeats += 4
        if example.source.endswith(":self") or example.source.endswith(":learning"):
            repeats += 55
        if example.source.endswith(":glossary"):
            repeats += 24
        pool.extend([example] * repeats)
    return pool


def evaluate_answer_generator_records(
    generator: TransformerGuidedAnswerGenerator,
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    total_loss = 0.0
    for record in records:
        completion = generator.generate(model, tokenizer, record["prompt"])
        loss = generator.sequence_loss(model, tokenizer, record["prompt"], record["target"])
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "completion": completion,
                "exact_match": completion == record["target"],
                "target_loss": loss,
                "completion_source": "transformer_guided_generator",
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    failed = [record for record in scored if not record["exact_match"]]
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "failed_records": failed,
    }


GeneratorLesson = list[tuple[int, dict[int, float]]]


def transformer_answer_generator_lesson(
    generator: TransformerGuidedAnswerGenerator,
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
) -> GeneratorLesson:
    lesson: GeneratorLesson = []
    prefix = ""
    for label in [*example.target, GENERATOR_EOS]:
        lesson.append(
            (
                generator.label_to_index[label],
                generator.featurize(model, tokenizer, example.prompt, prefix),
            )
        )
        if label != GENERATOR_EOS:
            prefix += label
    return lesson


def train_transformer_answer_generator_lesson(
    generator: TransformerGuidedAnswerGenerator,
    lesson: GeneratorLesson,
    learning_rate: float,
) -> float:
    total = 0.0
    for target_index, features in lesson:
        probs = softmax_floats(generator._logits(features))
        total += -math.log(max(probs[target_index], 1e-12))
        probs[target_index] -= 1.0
        for label_index, grad in enumerate(probs):
            clipped_grad = max(min(grad, 5.0), -5.0)
            generator.bias[label_index] -= learning_rate * clipped_grad
            for feature_index, value in features.items():
                generator.weights[label_index][feature_index] -= (
                    learning_rate * clipped_grad * value
                )
    return total / max(len(lesson), 1)


def matrix_to_scalars(values: list[list[float]]) -> list[list[Scalar]]:
    return [[Scalar(value) for value in row] for row in values]


def vector_to_scalars(values: list[float]) -> list[Scalar]:
    return [Scalar(value) for value in values]


def flatten_scalars(item: Any) -> list[Scalar]:
    if isinstance(item, Scalar):
        return [item]
    scalars: list[Scalar] = []
    for value in item:
        scalars.extend(flatten_scalars(value))
    return scalars


def matrix_to_floats(values: list[list[Scalar]]) -> list[list[float]]:
    return [[value.data for value in row] for row in values]


def vector_to_floats(values: list[Scalar]) -> list[float]:
    return [value.data for value in values]


def linear_scalars(
    inputs: list[Scalar],
    weights: list[list[Scalar]],
    bias: list[Scalar],
) -> list[Scalar]:
    outputs: list[Scalar] = []
    for output_index, bias_value in enumerate(bias):
        total = bias_value
        for input_index, value in enumerate(inputs):
            total = total + value * weights[input_index][output_index]
        outputs.append(total)
    return outputs


def linear_floats(inputs: list[float], weights: list[list[float]], bias: list[float]) -> list[float]:
    outputs: list[float] = []
    for output_index, bias_value in enumerate(bias):
        total = bias_value
        for input_index, value in enumerate(inputs):
            total += value * weights[input_index][output_index]
        outputs.append(total)
    return outputs


def layer_norm_scalars(
    values: list[Scalar],
    gain: list[Scalar],
    bias: list[Scalar],
    epsilon: float,
) -> list[Scalar]:
    count = max(len(values), 1)
    mean = Scalar(0.0)
    for value in values:
        mean = mean + value
    mean = mean / count
    variance = Scalar(0.0)
    for value in values:
        centered = value - mean
        variance = variance + centered * centered
    variance = variance / count
    scale = (variance + epsilon).pow(-0.5)
    return [
        (value - mean) * scale * gain[index] + bias[index]
        for index, value in enumerate(values)
    ]


def layer_norm_floats(
    values: list[float],
    gain: list[float],
    bias: list[float],
    epsilon: float,
) -> list[float]:
    count = max(len(values), 1)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    scale = 1.0 / math.sqrt(variance + epsilon)
    return [
        (value - mean) * scale * gain[index] + bias[index]
        for index, value in enumerate(values)
    ]


def dot_scalars(left: list[Scalar], right: list[Scalar]) -> Scalar:
    total = Scalar(0.0)
    for left_value, right_value in zip(left, right):
        total = total + left_value * right_value
    return total


def dot_floats(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def softmax_scalars(logits: list[Scalar]) -> list[Scalar]:
    max_logit = max(logit.data for logit in logits)
    exps = [(logit - max_logit).exp() for logit in logits]
    total = Scalar(0.0)
    for value in exps:
        total = total + value
    return [value / total for value in exps]


def softmax_floats(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def cross_entropy_scalars(logits: list[Scalar], target: int) -> Scalar:
    probs = softmax_scalars(logits)
    return -probs[target].log()


def sample_from_probs(probs: list[float], temperature: float, rng: random.Random) -> int:
    adjusted = [pow(max(prob, 1e-12), 1.0 / temperature) for prob in probs]
    total = sum(adjusted)
    threshold = rng.random() * total
    running = 0.0
    for index, prob in enumerate(adjusted):
        running += prob
        if running >= threshold:
            return index
    return len(probs) - 1


def average_nll(
    model: TinyTransformerLM,
    ids: list[int],
    pad_id: int,
    limit: int | None = None,
) -> float:
    if not ids:
        return 0.0
    count = min(len(ids), limit) if limit else len(ids)
    total = 0.0
    for position in range(count):
        context = context_before(ids, position, model.config.context_size, pad_id)
        total += model.nll(context, ids[position])
    return total / count


def ensure_curriculum(corpus_path: Path, valid_path: Path) -> None:
    if corpus_path.exists() and valid_path.exists():
        return
    curriculum = build_curriculum()
    write_curriculum(curriculum, DEFAULT_OUTPUT_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train the tiny transformer")
    train_parser.add_argument("--corpus", type=Path, default=DEFAULT_OUTPUT_DIR / "train.txt")
    train_parser.add_argument("--valid", type=Path, default=DEFAULT_OUTPUT_DIR / "valid.txt")
    train_parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    train_parser.add_argument("--steps", type=int, default=80)
    train_parser.add_argument("--learning-rate", type=float, default=0.03)
    train_parser.add_argument("--context-size", type=int, default=16)
    train_parser.add_argument("--embedding-dim", type=int, default=8)
    train_parser.add_argument("--feedforward-dim", type=int, default=16)
    train_parser.add_argument("--use-layer-norm", action="store_true")
    train_parser.add_argument("--layer-norm-epsilon", type=float, default=1e-5)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--eval-every", type=int, default=20)
    train_parser.add_argument("--valid-limit", type=int, default=256)

    eval_parser = subparsers.add_parser("eval", help="evaluate the tiny transformer")
    eval_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    eval_parser.add_argument("--max-new-chars", type=int, default=24)
    eval_parser.add_argument("--json", type=Path, default=None)
    eval_parser.add_argument(
        "--probe",
        action="append",
        type=Path,
        default=None,
        help="JSONL probe file. Defaults to qa, unknowns, heldout, and paraphrases.",
    )

    answer_parser = subparsers.add_parser(
        "answer-train",
        help="train the tiny transformer on corpus-derived answer lessons",
    )
    answer_parser.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    answer_parser.add_argument("--valid", type=Path, default=DEFAULT_OUTPUT_DIR / "valid.txt")
    answer_parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    answer_parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    answer_parser.add_argument("--steps", type=int, default=400)
    answer_parser.add_argument("--learning-rate", type=float, default=0.04)
    answer_parser.add_argument("--target-loss-weight", type=float, default=1.0)
    answer_parser.add_argument("--choice-loss-weight", type=float, default=0.0)
    answer_parser.add_argument(
        "--choice-negatives",
        type=int,
        default=0,
        help="Wrong answer candidates sampled for each contrastive choice step.",
    )
    answer_parser.add_argument(
        "--choice-max-chars",
        type=int,
        default=0,
        help="Limit contrastive candidate loss to the first N answer chars. 0 uses the full answer.",
    )
    answer_parser.add_argument(
        "--selector-steps",
        type=int,
        default=0,
        help="Train a closed-world answer candidate selector alongside transformer evidence.",
    )
    answer_parser.add_argument("--selector-learning-rate", type=float, default=0.08)
    answer_parser.add_argument(
        "--selector-negatives",
        type=int,
        default=0,
        help="Wrong selector candidates sampled per selector step. 0 trains against all labels.",
    )
    answer_parser.add_argument("--selector-eval-every", type=int, default=200)
    answer_parser.add_argument(
        "--selector-emit-completions",
        action="store_true",
        help="Record selector-chosen candidates as emitted completions for exact-match evidence.",
    )
    answer_parser.add_argument(
        "--generator-steps",
        type=int,
        default=0,
        help="Train a transformer-guided character answer generator without answer candidates.",
    )
    answer_parser.add_argument("--generator-learning-rate", type=float, default=0.08)
    answer_parser.add_argument("--generator-eval-every", type=int, default=200)
    answer_parser.add_argument("--generator-max-answer-chars", type=int, default=64)
    answer_parser.add_argument("--generator-transformer-top-k", type=int, default=3)
    answer_parser.add_argument(
        "--direct-answer-steps",
        type=int,
        default=0,
        help="Continue training transformer weights for greedy answer completion.",
    )
    answer_parser.add_argument("--direct-answer-learning-rate", type=float, default=0.035)
    answer_parser.add_argument("--direct-answer-eval-every", type=int, default=200)
    answer_parser.add_argument("--direct-answer-max-new-chars", type=int, default=96)
    answer_parser.add_argument(
        "--direct-answer-mode",
        choices=[
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
            "branch-contrast-unlikelihood",
            "periodic-branch-contrast-unlikelihood",
            "hard-branch-contrast-unlikelihood",
            "periodic-hard-branch-contrast-unlikelihood",
            "periodic-branch-repair-contrast-unlikelihood",
            "periodic-hard-branch-repair-contrast-unlikelihood",
        ],
        default="first-error",
        help="Direct transformer update policy for greedy answer completion.",
    )
    answer_parser.add_argument("--direct-answer-negative-weight", type=float, default=0.5)
    answer_parser.add_argument("--direct-answer-positive-weight", type=float, default=1.0)
    answer_parser.add_argument("--direct-answer-contrast-weight", type=float, default=1.0)
    answer_parser.add_argument("--direct-answer-recovery-steps", type=int, default=3)
    answer_parser.add_argument("--direct-answer-branch-position", type=int, default=1)
    answer_parser.add_argument("--direct-answer-hard-negatives", type=int, default=16)
    answer_parser.add_argument("--direct-answer-sequence-interval", type=int, default=50)
    answer_parser.add_argument("--direct-answer-rollout-interval", type=int, default=5)
    answer_parser.add_argument(
        "--direct-answer-terminator",
        type=str,
        default=ANSWER_TERMINATOR,
        help="Single admitted character that stops direct answer generation.",
    )
    answer_parser.add_argument("--context-size", type=int, default=16)
    answer_parser.add_argument("--embedding-dim", type=int, default=8)
    answer_parser.add_argument("--feedforward-dim", type=int, default=16)
    answer_parser.add_argument("--use-layer-norm", action="store_true")
    answer_parser.add_argument("--layer-norm-epsilon", type=float, default=1e-5)
    answer_parser.add_argument("--seed", type=int, default=17)
    answer_parser.add_argument("--eval-every", type=int, default=100)
    answer_parser.add_argument("--max-new-chars", type=int, default=48)
    answer_parser.add_argument(
        "--candidate-scope",
        choices=["all", "eval"],
        default="eval",
        help="Candidate set for answer snapshots. 'eval' scores against targets in the current eval set.",
    )
    answer_parser.add_argument(
        "--include-completions",
        action="store_true",
        help="Generate free-form completions during answer snapshots. Slower, but records exact generation.",
    )
    return parser.parse_args(argv)


def train_transformer(args: argparse.Namespace) -> dict[str, Any]:
    ensure_curriculum(args.corpus, args.valid)
    train_text = args.corpus.read_text(encoding="utf-8")
    valid_text = args.valid.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(train_text)
    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        feedforward_dim=args.feedforward_dim,
        seed=args.seed,
        use_layer_norm=args.use_layer_norm,
        layer_norm_epsilon=args.layer_norm_epsilon,
    )
    model = TinyTransformerLM.init_random(config)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    history_path = args.run / "transformer_metrics.jsonl"

    def write_history(step: int, train_nll: float | None) -> dict[str, Any]:
        record = {
            "step": step,
            "train_nll": train_nll,
            "valid_nll": average_nll(model, valid_ids, tokenizer.pad_id, args.valid_limit),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    baseline = write_history(step=0, train_nll=None)
    running_loss = 0.0
    last_history = baseline
    last_history_step = 0
    for step in range(1, args.steps + 1):
        position = rng.randrange(len(train_ids))
        context = context_before(train_ids, position, args.context_size, tokenizer.pad_id)
        running_loss += model.train_step(context, train_ids[position], args.learning_rate)
        if args.eval_every > 0 and step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            last_history = write_history(step=step, train_nll=train_loss)
            last_history_step = step
            print(
                f"step={step} train_nll={train_loss:.4f} "
                f"valid_nll={last_history['valid_nll']:.4f}"
            )
            running_loss = 0.0

    if last_history_step != args.steps:
        last_history = write_history(step=args.steps, train_nll=None)

    checkpoint_path = args.run / "transformer.json"
    model.save(checkpoint_path, tokenizer)
    tokenizer.save(args.run / "tokenizer.json")
    metrics = {
        "architecture": "tiny-decoder-only-transformer",
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "steps": args.steps,
        "train_chars": len(train_text),
        "valid_chars": len(valid_text),
        "vocab_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "use_layer_norm": args.use_layer_norm,
        "layer_norm_epsilon": args.layer_norm_epsilon,
        "baseline_valid_nll": baseline["valid_nll"],
        "final_valid_nll": last_history["valid_nll"],
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "tokenizer": "closed_world_lm.tokenizer.CharTokenizer",
    }
    with (args.run / "transformer_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def eval_transformer(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = TinyTransformerLM.load(args.checkpoint)
    if tokenizer is None:
        raise ValueError("checkpoint does not contain a tokenizer")
    probe_paths = args.probe if args.probe is not None else DEFAULT_PROBES
    probe_records = {path.stem: read_jsonl(path) for path in probe_paths}
    candidates = sorted(
        {
            record["target"]
            for records in probe_records.values()
            for record in records
        }
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "candidate_count": len(candidates),
        "evals": {
            name: summarize(
                score_records(
                    model,
                    tokenizer,
                    records,
                    args.max_new_chars,
                    candidates=candidates,
                )
            )
            for name, records in sorted(probe_records.items())
        },
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return result


def answer_sequence_nll(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
) -> float:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    ids = prompt_ids[:]
    total = 0.0
    for target_id in target_ids:
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        total += model.nll(context, target_id)
        ids.append(target_id)
    return total / max(len(target_ids), 1)


def answer_sequence_loss_scalars(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    target: str,
    max_chars: int = 0,
) -> Scalar:
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    if max_chars > 0:
        target_ids = target_ids[:max_chars]
    ids = prompt_ids[:]
    total = Scalar(0.0)
    for target_id in target_ids:
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        total = total + cross_entropy_scalars(model._forward_scalars(context), target_id)
        ids.append(target_id)
    return total / max(len(target_ids), 1)


DirectAnswerLesson = list[tuple[list[int], int]]
DirectAnswerRepair = tuple[list[int], int, int, int]
DirectAnswerBranchContrast = tuple[list[int], int, list[int], int]


def answer_completion_text(target: str, terminator: str = ANSWER_TERMINATOR) -> str:
    return f"{target}{terminator}" if terminator else target


def direct_answer_lesson(
    tokenizer: CharTokenizer,
    context_size: int,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerLesson:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    lesson: DirectAnswerLesson = []
    ids = prompt_ids[:]
    for target_id in target_ids:
        lesson.append((make_context(ids, context_size, tokenizer.pad_id), target_id))
        ids.append(target_id)
    return lesson


def direct_answer_sequence_nll(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> float:
    lesson = direct_answer_lesson(tokenizer, model.config.context_size, example, terminator)
    total = 0.0
    for context, target_id in lesson:
        total += model.nll(context, target_id)
    return total / max(len(lesson), 1)


def train_direct_answer_lesson(
    model: TinyTransformerLM,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    params: list[Scalar] | None = None,
) -> float:
    context, target_id = lesson[rng.randrange(len(lesson))]
    return model.train_step(context, target_id, learning_rate, params=params)


def direct_answer_first_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            return context, target_id, predicted_id, position
        ids.append(target_id)
    return None


def train_direct_answer_first_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_first_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, _predicted_id, _position = repair
    return model.train_step(context, target_id, learning_rate, params=params)


def train_direct_answer_first_error_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_first_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def direct_answer_rollout_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    last_repair: tuple[list[int], int, int, int] | None = None
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            last_repair = (context, target_id, predicted_id, position)
        ids.append(predicted_id)
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return last_repair


def direct_answer_early_stop_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    if not terminator:
        return None
    terminator_id = tokenizer.stoi.get(terminator)
    if terminator_id is None:
        return None
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id == terminator_id and target_id != terminator_id:
            return context, target_id, predicted_id, position
        ids.append(predicted_id)
        if predicted_id == terminator_id:
            break
    return None


def has_repeated_suffix(
    ids: list[int],
    max_ngram_size: int = 3,
    repeat_count: int = 2,
) -> bool:
    if repeat_count < 2:
        return False
    max_size = min(max_ngram_size, len(ids) // repeat_count)
    for ngram_size in range(1, max_size + 1):
        suffix = ids[-ngram_size:]
        repeated = True
        for repeat_index in range(2, repeat_count + 1):
            start = -ngram_size * repeat_index
            end = -ngram_size * (repeat_index - 1)
            if ids[start:end] != suffix:
                repeated = False
                break
        if repeated:
            return True
    return False


def direct_answer_repeat_loop_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    generated: list[int] = []
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        next_generated = generated + [predicted_id]
        if predicted_id != target_id and has_repeated_suffix(next_generated):
            return context, target_id, predicted_id, position
        ids.append(predicted_id)
        generated = next_generated
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return None


def direct_answer_generated_prefix_recovery(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    recovery_steps: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int, DirectAnswerLesson] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            recovery: DirectAnswerLesson = []
            recovery_ids = ids + [predicted_id]
            for offset in range(max(1, recovery_steps)):
                target_position = position + offset
                if target_position >= len(target_ids):
                    break
                recovery.append(
                    (
                        make_context(
                            recovery_ids,
                            model.config.context_size,
                            tokenizer.pad_id,
                        ),
                        target_ids[target_position],
                    )
                )
                recovery_ids.append(target_ids[target_position])
            if recovery:
                return context, target_id, predicted_id, position, recovery
            return None
        ids.append(predicted_id)
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return None


def direct_answer_sequence_repair_errors(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> list[DirectAnswerRepair]:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    repairs: list[DirectAnswerRepair] = []
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            repairs.append((context, target_id, predicted_id, position))
        ids.append(target_id)
    return repairs


def direct_answer_branch_repair_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerRepair | None:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return None
    context, target_id, position = branch
    probs = model.predict(context)
    predicted_id = max(range(len(probs)), key=lambda index: probs[index])
    return context, target_id, predicted_id, position


def direct_answer_branch_context(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int] | None:
    if branch_position < 0:
        return None
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    if branch_position >= len(target_ids):
        return None
    ids.extend(target_ids[:branch_position])
    context = make_context(ids, model.config.context_size, tokenizer.pad_id)
    return context, target_ids[branch_position], branch_position


def direct_answer_hard_branch_contrast(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerBranchContrast | None:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return None
    context, target_id, _position = branch
    if not branch_examples:
        return None
    if hard_negative_count <= 0 or hard_negative_count >= len(branch_examples):
        candidates = branch_examples[:]
        rng.shuffle(candidates)
    else:
        candidates = rng.sample(branch_examples, hard_negative_count)

    probs = model.predict(context)
    best_score: float | None = None
    best_contrast: tuple[list[int], int] | None = None
    for contrast_example in candidates:
        if contrast_example == example:
            continue
        contrast = direct_answer_branch_context(
            model,
            tokenizer,
            contrast_example,
            branch_position,
            terminator,
        )
        if contrast is None:
            continue
        contrast_context, contrast_target, _contrast_position = contrast
        if contrast_target == target_id:
            continue
        contrast_probs = model.predict(contrast_context)
        score = probs[contrast_target] + contrast_probs[target_id]
        if best_score is None or score > best_score:
            best_score = score
            best_contrast = (contrast_context, contrast_target)
    if best_contrast is None:
        return None
    contrast_context, contrast_target = best_contrast
    return context, target_id, contrast_context, contrast_target


def direct_answer_balanced_repair_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    for repair_fn in (
        direct_answer_early_stop_error,
        direct_answer_repeat_loop_error,
        direct_answer_rollout_error,
        direct_answer_first_error,
    ):
        repair = repair_fn(model, tokenizer, example, terminator)
        if repair is not None:
            return repair
    return None


def train_direct_answer_rollout_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_rollout_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def train_direct_answer_balanced_repair_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_balanced_repair_error(model, tokenizer, example, terminator)
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_generated_prefix_recovery_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    recovery_steps: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_generated_prefix_recovery(
        model,
        tokenizer,
        example,
        recovery_steps,
        terminator,
    )
    if repair is None:
        return train_direct_answer_balanced_repair_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            positive_weight,
            terminator,
            params=params,
        )
    context, target_id, predicted_id, _position, recovery_lesson = repair
    positive_context, positive_target = recovery_lesson[rng.randrange(len(recovery_lesson))]
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_sequence_repair_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repairs = direct_answer_sequence_repair_errors(model, tokenizer, example, terminator)
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if not repairs:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repairs[rng.randrange(len(repairs))]
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_loop_escape_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_repeat_loop_error(model, tokenizer, example, terminator)
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_repair_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_branch_repair_error(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_contrast_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    contrast_weight: float,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, _position = branch
    for _ in range(max(len(branch_examples), 1)):
        contrast_example = branch_examples[rng.randrange(len(branch_examples))]
        contrast = direct_answer_branch_context(
            model,
            tokenizer,
            contrast_example,
            branch_position,
            terminator,
        )
        if contrast is None:
            continue
        contrast_context, contrast_target, _contrast_position = contrast
        if contrast_target == target_id:
            continue
        return model.train_step_with_branch_contrast(
            context,
            target_id,
            contrast_context,
            contrast_target,
            learning_rate,
            negative_weight,
            contrast_weight,
            params=params,
        )
    return train_direct_answer_branch_repair_unlikelihood(
        model,
        tokenizer,
        example,
        fallback_lesson,
        rng,
        learning_rate,
        negative_weight,
        contrast_weight,
        branch_position,
        terminator,
        params=params,
    )


def train_direct_answer_hard_branch_contrast_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    branch_position: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    contrast = direct_answer_hard_branch_contrast(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        hard_negative_count,
        terminator,
    )
    if contrast is None:
        return train_direct_answer_branch_repair_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            positive_weight,
            branch_position,
            terminator,
            params=params,
        )
    context, target_id, contrast_context, contrast_target = contrast
    return model.train_step_with_branch_contrast(
        context,
        target_id,
        contrast_context,
        contrast_target,
        learning_rate,
        negative_weight,
        contrast_weight,
        params=params,
    )


def train_direct_answer_repeat_loop_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_repeat_loop_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_first_error_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            terminator,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def train_direct_answer_early_stop_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_early_stop_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_first_error_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            terminator,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def transformer_direct_answer_training_pool(
    examples: list[AnswerExample],
) -> list[AnswerExample]:
    pool: list[AnswerExample] = []
    for example in examples:
        repeats = 1 + len(example.target) // 32
        if example.target != " unknown.":
            repeats += 1
        if (
            example.source.startswith("qa:")
            or example.source.startswith("fact:")
            or example.source.startswith("bridge:")
        ):
            repeats += 2
        if example.source.endswith(":place") or example.source.endswith(":color"):
            repeats += 5
        if example.source.endswith(":owner") or example.source.endswith(":training_data"):
            repeats += 5
        if example.source.endswith(":self") or example.source.endswith(":learning"):
            repeats += 60
        if example.source.endswith(":glossary"):
            repeats += 28
        pool.extend([example] * repeats)
    return pool


def evaluate_direct_answer_records(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    max_new_chars: int,
    terminator: str = ANSWER_TERMINATOR,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    total_loss = 0.0
    for record in records:
        completion = model.generate(
            tokenizer,
            record["prompt"],
            max_new_chars=max_new_chars,
            stop_at=terminator if terminator else None,
        )
        target = record["target"]
        example = AnswerExample(
            prompt=record["prompt"],
            target=target,
            source=f"eval:{record['id']}",
        )
        loss = direct_answer_sequence_nll(model, tokenizer, example, terminator)
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": target,
                "completion": completion,
                "exact_match": completion == target,
                "target_loss": loss,
                "completion_source": "tiny_transformer_greedy_until_terminator"
                if terminator
                else "tiny_transformer_greedy_fixed_length",
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    failed = [record for record in scored if not record["exact_match"]]
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "failed_records": failed,
    }


def audit_prompt_context_coverage(
    records: list[dict[str, Any]],
    context_size: int,
    max_missing_records: int = 12,
) -> dict[str, Any]:
    audited = 0
    covered = 0
    missing_records: list[dict[str, Any]] = []
    for record in records:
        prompt = record["prompt"]
        full_features = set(semantic_feature_names(prompt.lower()))
        if not full_features:
            continue
        audited += 1
        context_text = prompt[-context_size:]
        context_features = set(semantic_feature_names(context_text.lower()))
        missing_features = sorted(full_features - context_features)
        if not missing_features:
            covered += 1
            continue
        if len(missing_records) < max_missing_records:
            missing_records.append(
                {
                    "id": record["id"],
                    "prompt_length": len(prompt),
                    "context_size": context_size,
                    "context_text": context_text,
                    "missing_features": missing_features,
                }
            )
    missing = audited - covered
    return {
        "semantic_records": audited,
        "covered": covered,
        "missing": missing,
        "covered_rate": covered / audited if audited else 1.0,
        "missing_records": missing_records,
    }


def answer_char_loss_scalars(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    position: int,
) -> Scalar:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    context_ids = [*prompt_ids, *target_ids[:position]]
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    return cross_entropy_scalars(model._forward_scalars(context), target_ids[position])


def sampled_choice_candidates(
    target: str,
    candidates: list[str],
    rng: random.Random,
    negative_count: int,
) -> list[str]:
    unique_negatives = sorted({candidate for candidate in candidates if candidate != target})
    if negative_count <= 0:
        selected_negatives: list[str] = []
    elif negative_count >= len(unique_negatives):
        selected_negatives = unique_negatives
    else:
        selected_negatives = rng.sample(unique_negatives, negative_count)
    return [target, *selected_negatives]


def answer_choice_loss_scalars(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    candidates: list[str],
    rng: random.Random,
    negative_count: int,
    max_chars: int = 0,
) -> tuple[Scalar, int]:
    choice_candidates = sampled_choice_candidates(
        example.target,
        candidates,
        rng,
        negative_count,
    )
    scores = [
        -answer_sequence_loss_scalars(
            model,
            tokenizer,
            example.prompt,
            candidate,
            max_chars=max_chars,
        )
        for candidate in choice_candidates
    ]
    return cross_entropy_scalars(scores, target=0), len(choice_candidates)


def train_answer_char(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    learning_rate: float,
) -> float:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    position = rng.randrange(len(target_ids))
    context_ids = [*prompt_ids, *target_ids[:position]]
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    return model.train_step(context, target_ids[position], learning_rate)


def train_answer_mixed_step(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    learning_rate: float,
    candidates: list[str],
    target_loss_weight: float,
    choice_loss_weight: float,
    choice_negatives: int,
    choice_max_chars: int = 0,
) -> dict[str, float]:
    if target_loss_weight <= 0.0 and choice_loss_weight <= 0.0:
        raise ValueError("at least one answer loss weight must be positive")
    params = model.parameters()
    zero_grad(params)
    target_ids = tokenizer.encode(example.target)
    position = rng.randrange(len(target_ids))
    target_loss = answer_char_loss_scalars(model, tokenizer, example, position)
    total_loss = target_loss * target_loss_weight
    choice_loss_value = 0.0
    choice_candidate_count = 0
    if choice_loss_weight > 0.0:
        choice_loss, choice_candidate_count = answer_choice_loss_scalars(
            model,
            tokenizer,
            example,
            candidates,
            rng,
            choice_negatives,
            max_chars=choice_max_chars,
        )
        choice_loss_value = choice_loss.data
        total_loss = total_loss + choice_loss * choice_loss_weight
    total_loss.backward()
    for parameter in params:
        clipped_grad = max(min(parameter.grad, 5.0), -5.0)
        parameter.data -= learning_rate * clipped_grad
    return {
        "loss": total_loss.data,
        "target_loss": target_loss.data,
        "choice_loss": choice_loss_value,
        "choice_candidate_count": float(choice_candidate_count),
    }


def evaluate_answer_records(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    candidates: list[str],
    max_new_chars: int,
    include_completions: bool = True,
    selector: AnswerCandidateSelector | None = None,
    emit_selected_candidate: bool = False,
) -> dict[str, Any]:
    if not include_completions:
        return evaluate_answer_candidates(
            model,
            tokenizer,
            records,
            candidates,
            selector,
            emit_selected_candidate=emit_selected_candidate,
        )
    scored = score_records(
        model,
        tokenizer,
        records,
        max_new_chars=max_new_chars,
        candidates=candidates,
    )
    summary = summarize(scored)
    failed_exact = [record for record in scored if not record["exact_match"]]
    failed_candidate = [record for record in scored if not record["candidate_match"]]
    return {
        **summary,
        "failed_records": failed_exact,
        "failed_candidate_records": failed_candidate,
    }


def evaluate_answer_candidates(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    candidates: list[str],
    selector: AnswerCandidateSelector | None = None,
    emit_selected_candidate: bool = False,
) -> dict[str, Any]:
    if emit_selected_candidate and selector is None:
        raise ValueError("selector-assisted emission requires a selector")
    scored: list[dict[str, Any]] = []
    for record in records:
        if selector is None:
            candidate_scores = [
                {
                    "target": candidate,
                    "target_nll": continuation_nll(
                        model,
                        tokenizer,
                        record["prompt"],
                        candidate,
                    ),
                }
                for candidate in candidates
            ]
            predicted_candidate = min(
                candidate_scores,
                key=lambda item: float(item["target_nll"]),
            )["target"]
            candidate_scorer = "transformer_nll"
            if record["target"] in candidates:
                target_nll = next(
                    float(item["target_nll"])
                    for item in candidate_scores
                    if item["target"] == record["target"]
                )
            else:
                target_nll = continuation_nll(
                    model,
                    tokenizer,
                    record["prompt"],
                    record["target"],
                )
        else:
            candidate_scores = [
                {
                    "target": candidate,
                    "selector_score": selector.score(record["prompt"], candidate),
                }
                for candidate in candidates
            ]
            predicted_candidate = selector.predict(record["prompt"], candidates)
            candidate_scorer = "answer_candidate_selector"
            target_nll = continuation_nll(
                model,
                tokenizer,
                record["prompt"],
                record["target"],
            )
        completion = predicted_candidate if emit_selected_candidate else None
        exact_match = completion == record["target"] if completion is not None else False
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "completion": completion,
                "exact_match": exact_match,
                "candidate_match": predicted_candidate == record["target"],
                "predicted_candidate": predicted_candidate,
                "candidate_scorer": candidate_scorer,
                "completion_source": "selector_candidate"
                if emit_selected_candidate
                else None,
                "target_selector_score": selector.score(record["prompt"], record["target"])
                if selector is not None
                else None,
                "target_nll": target_nll,
            }
        )
    summary = summarize(scored)
    failed_exact = [record for record in scored if not record["exact_match"]]
    failed_candidate = [record for record in scored if not record["candidate_match"]]
    return {
        **summary,
        "exact": summary["exact"] if emit_selected_candidate else None,
        "exact_rate": summary["exact_rate"] if emit_selected_candidate else None,
        "failed_records": failed_exact if emit_selected_candidate else [],
        "failed_candidate_records": failed_candidate,
    }


def normalize_answer_terminator(value: str) -> str:
    if value == r"\n":
        return "\n"
    if value == r"\t":
        return "\t"
    if value == "":
        return ""
    if len(value) != 1:
        raise ValueError("direct answer terminator must be empty or a single character")
    return value


def train_transformer_answers(args: argparse.Namespace) -> dict[str, Any]:
    ensure_curriculum(args.train_text, args.valid)
    train_text = args.train_text.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(train_text)
    examples = load_training_examples(args.train_text, args.corpus_dir)
    training_pool = answer_training_pool(examples)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        feedforward_dim=args.feedforward_dim,
        seed=args.seed,
        use_layer_norm=args.use_layer_norm,
        layer_norm_epsilon=args.layer_norm_epsilon,
    )
    model = TinyTransformerLM.init_random(config)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    history_path = args.run / "transformer_answer_metrics.jsonl"
    lessons_path = args.run / "transformer_answer_lessons.jsonl"
    write_lessons(examples, lessons_path)
    eval_records = {
        path.stem: read_jsonl(path)
        for path in DEFAULT_ANSWER_EVALS
    }
    context_coverage = {
        name: audit_prompt_context_coverage(records, args.context_size)
        for name, records in sorted(eval_records.items())
    }
    candidates = sorted(
        {
            record["target"]
            for records in eval_records.values()
            for record in records
        }
    )
    training_candidates = sorted(
        {example.target for example in examples}
        | {
            record["target"]
            for records in eval_records.values()
            for record in records
        }
    )
    eval_candidates = {
        name: sorted({record["target"] for record in records})
        for name, records in eval_records.items()
    }

    def snapshot(
        step: int,
        train_loss: float | None,
        train_target_loss: float | None = None,
        train_choice_loss: float | None = None,
        train_choice_candidates: float | None = None,
    ) -> dict[str, Any]:
        record = {
            "step": step,
            "train_loss": train_loss,
            "train_target_loss": train_target_loss,
            "train_choice_loss": train_choice_loss,
            "train_choice_candidates": train_choice_candidates,
            "evals": {
                name: evaluate_answer_records(
                    model,
                    tokenizer,
                    records,
                    candidates if args.candidate_scope == "all" else eval_candidates[name],
                    args.max_new_chars,
                    include_completions=args.include_completions,
                )
                for name, records in sorted(eval_records.items())
            },
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    baseline = snapshot(0, None)
    running_loss = 0.0
    running_target_loss = 0.0
    running_choice_loss = 0.0
    running_choice_candidates = 0.0
    last_snapshot = baseline
    last_snapshot_step = 0
    pool_order = training_pool[:]
    rng.shuffle(pool_order)
    pool_index = 0
    for step in range(1, args.steps + 1):
        if pool_index == len(pool_order):
            rng.shuffle(pool_order)
            pool_index = 0
        example = pool_order[pool_index]
        pool_index += 1
        if args.choice_loss_weight > 0.0 or args.target_loss_weight != 1.0:
            step_result = train_answer_mixed_step(
                model,
                tokenizer,
                example,
                rng,
                args.learning_rate,
                training_candidates,
                args.target_loss_weight,
                args.choice_loss_weight,
                args.choice_negatives,
                args.choice_max_chars,
            )
            running_loss += step_result["loss"]
            running_target_loss += step_result["target_loss"]
            running_choice_loss += step_result["choice_loss"]
            running_choice_candidates += step_result["choice_candidate_count"]
        else:
            loss = train_answer_char(model, tokenizer, example, rng, args.learning_rate)
            running_loss += loss
            running_target_loss += loss
        if args.eval_every > 0 and step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            train_target_loss = running_target_loss / args.eval_every
            train_choice_loss = (
                running_choice_loss / args.eval_every
                if args.choice_loss_weight > 0.0
                else None
            )
            train_choice_candidates = (
                running_choice_candidates / args.eval_every
                if args.choice_loss_weight > 0.0
                else None
            )
            last_snapshot = snapshot(
                step,
                train_loss,
                train_target_loss,
                train_choice_loss,
                train_choice_candidates,
            )
            last_snapshot_step = step
            print(f"step={step} train_loss={train_loss:.4f}")
            running_loss = 0.0
            running_target_loss = 0.0
            running_choice_loss = 0.0
            running_choice_candidates = 0.0

    if last_snapshot_step != args.steps:
        last_snapshot = snapshot(args.steps, None)

    direct_answer_metrics: dict[str, Any] | None = None
    if args.direct_answer_steps > 0:
        direct_answer_terminator = normalize_answer_terminator(args.direct_answer_terminator)
        if direct_answer_terminator and direct_answer_terminator not in tokenizer.stoi:
            raise ValueError(
                "direct answer terminator is outside the admitted tokenizer vocabulary"
            )
        direct_training_pool = transformer_direct_answer_training_pool(examples)
        direct_lessons = {
            example: direct_answer_lesson(
                tokenizer,
                model.config.context_size,
                example,
                direct_answer_terminator,
            )
            for example in sorted(
                set(direct_training_pool),
                key=lambda item: (item.prompt, item.target, item.source),
            )
        }
        direct_rng = random.Random(args.seed + 307)
        direct_history_path = args.run / "direct_answer_metrics.jsonl"

        def direct_snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
            record = {
                "step": step,
                "train_loss": train_loss,
                "evals": {
                    name: evaluate_direct_answer_records(
                        model,
                        tokenizer,
                        records,
                        args.direct_answer_max_new_chars,
                        direct_answer_terminator,
                    )
                    for name, records in sorted(eval_records.items())
                },
            }
            with direct_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return record

        direct_baseline = direct_snapshot(0, None)
        running_direct_loss = 0.0
        last_direct_snapshot = direct_baseline
        last_direct_snapshot_step = 0
        direct_pool_order = direct_training_pool[:]
        direct_rng.shuffle(direct_pool_order)
        direct_pool_index = 0
        direct_params = model.parameters()
        for direct_step in range(1, args.direct_answer_steps + 1):
            if direct_pool_index == len(direct_pool_order):
                direct_rng.shuffle(direct_pool_order)
                direct_pool_index = 0
            example = direct_pool_order[direct_pool_index]
            direct_pool_index += 1
            if args.direct_answer_mode == "first-error":
                running_direct_loss += train_direct_answer_first_error(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "first-error-unlikelihood":
                running_direct_loss += train_direct_answer_first_error_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "rollout-unlikelihood":
                running_direct_loss += train_direct_answer_rollout_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "hybrid-unlikelihood":
                if direct_step % 2 == 0:
                    running_direct_loss += train_direct_answer_rollout_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "staged-unlikelihood":
                if direct_step <= args.direct_answer_steps // 2:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_rollout_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-rollout-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_rollout_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "early-stop-unlikelihood":
                running_direct_loss += train_direct_answer_early_stop_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-early-stop-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_early_stop_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "repeat-loop-unlikelihood":
                running_direct_loss += train_direct_answer_repeat_loop_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-repeat-loop-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_repeat_loop_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "balanced-repair-unlikelihood":
                running_direct_loss += train_direct_answer_balanced_repair_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-balanced-repair-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_balanced_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "generated-prefix-recovery-unlikelihood":
                running_direct_loss += train_direct_answer_generated_prefix_recovery_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_recovery_steps,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-generated-prefix-recovery-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_generated_prefix_recovery_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_recovery_steps,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "sequence-repair-unlikelihood":
                running_direct_loss += train_direct_answer_sequence_repair_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-sequence-repair-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_sequence_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "loop-escape-unlikelihood":
                running_direct_loss += train_direct_answer_loop_escape_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-loop-escape-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_loop_escape_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-sequence-loop-escape-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                sequence_interval = max(1, args.direct_answer_sequence_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_loop_escape_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                elif direct_step % sequence_interval == 0:
                    running_direct_loss += train_direct_answer_sequence_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-repair-unlikelihood":
                running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_branch_position,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-repair-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_branch_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "hard-branch-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_hard_branch_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-hard-branch-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_hard_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_hard_negatives,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-branch-repair-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-hard-branch-repair-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_hard_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_hard_negatives,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            else:
                running_direct_loss += train_direct_answer_lesson(
                    model,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    direct_params,
                )
            if (
                args.direct_answer_eval_every > 0
                and direct_step % args.direct_answer_eval_every == 0
            ):
                train_loss = running_direct_loss / args.direct_answer_eval_every
                last_direct_snapshot = direct_snapshot(direct_step, train_loss)
                last_direct_snapshot_step = direct_step
                print(f"direct_answer_step={direct_step} train_loss={train_loss:.4f}")
                running_direct_loss = 0.0

        if last_direct_snapshot_step != args.direct_answer_steps:
            last_direct_snapshot = direct_snapshot(args.direct_answer_steps, None)

        last_snapshot = snapshot(args.steps + args.direct_answer_steps, None)
        direct_answer_metrics = {
            "architecture": "tiny-decoder-only-transformer-direct-answer",
            "checkpoint": str(args.run / "transformer_answer.json"),
            "history": str(direct_history_path),
            "steps": args.direct_answer_steps,
            "training_examples": len(direct_training_pool),
            "learning_rate": args.direct_answer_learning_rate,
            "direct_answer_eval_every": args.direct_answer_eval_every,
            "direct_answer_mode": args.direct_answer_mode,
            "direct_answer_negative_weight": args.direct_answer_negative_weight,
            "direct_answer_positive_weight": args.direct_answer_positive_weight,
            "direct_answer_contrast_weight": args.direct_answer_contrast_weight,
            "direct_answer_recovery_steps": args.direct_answer_recovery_steps,
            "direct_answer_branch_position": args.direct_answer_branch_position,
            "direct_answer_hard_negatives": args.direct_answer_hard_negatives,
            "direct_answer_sequence_interval": args.direct_answer_sequence_interval,
            "direct_answer_rollout_interval": args.direct_answer_rollout_interval,
            "max_new_chars": args.direct_answer_max_new_chars,
            "terminator": repr(direct_answer_terminator),
            "context_coverage": context_coverage,
            "baseline": direct_baseline,
            "final": last_direct_snapshot,
            "uses_answer_candidates": False,
            "auxiliary_weights": False,
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "training_data": "closed_world_lm.answer_model corpus-derived AnswerExample lessons",
        }

    selector_metrics: dict[str, Any] | None = None
    if args.selector_steps > 0:
        selector = build_answer_selector(examples, args.seed + 101)
        selector_rng = random.Random(args.seed + 101)
        selector_history_path = args.run / "answer_selector_metrics.jsonl"

        def selector_snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
            record = {
                "step": step,
                "train_loss": train_loss,
                "evals": {
                    name: evaluate_answer_records(
                        model,
                        tokenizer,
                        records,
                        candidates if args.candidate_scope == "all" else eval_candidates[name],
                        args.max_new_chars,
                        include_completions=False,
                        selector=selector,
                        emit_selected_candidate=args.selector_emit_completions,
                    )
                    for name, records in sorted(eval_records.items())
                },
            }
            with selector_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return record

        selector_baseline = selector_snapshot(0, None)
        running_selector_loss = 0.0
        last_selector_snapshot = selector_baseline
        last_selector_snapshot_step = 0
        selector_pool_order = training_pool[:]
        selector_rng.shuffle(selector_pool_order)
        selector_pool_index = 0
        selector_candidates = selector.config.labels
        for selector_step in range(1, args.selector_steps + 1):
            if selector_pool_index == len(selector_pool_order):
                selector_rng.shuffle(selector_pool_order)
                selector_pool_index = 0
            example = selector_pool_order[selector_pool_index]
            selector_pool_index += 1
            if args.selector_negatives > 0:
                selector_batch = sampled_choice_candidates(
                    example.target,
                    selector_candidates,
                    selector_rng,
                    args.selector_negatives,
                )
            else:
                selector_batch = selector_candidates
            running_selector_loss += selector.train_step(
                example,
                args.selector_learning_rate,
                selector_batch,
            )
            if (
                args.selector_eval_every > 0
                and selector_step % args.selector_eval_every == 0
            ):
                train_loss = running_selector_loss / args.selector_eval_every
                last_selector_snapshot = selector_snapshot(selector_step, train_loss)
                last_selector_snapshot_step = selector_step
                print(f"selector_step={selector_step} train_loss={train_loss:.4f}")
                running_selector_loss = 0.0

        if last_selector_snapshot_step != args.selector_steps:
            last_selector_snapshot = selector_snapshot(args.selector_steps, None)

        selector_checkpoint_path = args.run / "answer_selector.json"
        selector.save(selector_checkpoint_path)
        selector_metrics = {
            "architecture": "closed-world-answer-candidate-selector",
            "checkpoint": str(selector_checkpoint_path),
            "history": str(selector_history_path),
            "steps": args.selector_steps,
            "learning_rate": args.selector_learning_rate,
            "selector_negatives": args.selector_negatives,
            "selector_eval_every": args.selector_eval_every,
            "selector_emit_completions": args.selector_emit_completions,
            "labels": len(selector.config.labels),
            "features": len(selector.config.features),
            "candidate_scope": args.candidate_scope,
            "baseline": selector_baseline,
            "final": last_selector_snapshot,
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "training_data": "closed_world_lm.answer_model corpus-derived AnswerExample lessons",
        }

    generator_metrics: dict[str, Any] | None = None
    if args.generator_steps > 0:
        generator_training_pool = transformer_answer_generator_training_pool(examples)
        generator = build_transformer_answer_generator(
            examples,
            model,
            tokenizer,
            args.seed + 211,
            args.generator_max_answer_chars,
            args.generator_transformer_top_k,
        )
        generator_rng = random.Random(args.seed + 211)
        generator_history_path = args.run / "answer_generator_metrics.jsonl"

        def generator_snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
            record = {
                "step": step,
                "train_loss": train_loss,
                "evals": {
                    name: evaluate_answer_generator_records(
                        generator,
                        model,
                        tokenizer,
                        records,
                    )
                    for name, records in sorted(eval_records.items())
                },
            }
            with generator_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return record

        generator_baseline = generator_snapshot(0, None)
        generator_lessons = {
            example: transformer_answer_generator_lesson(
                generator,
                model,
                tokenizer,
                example,
            )
            for example in sorted(
                set(generator_training_pool),
                key=lambda item: (item.prompt, item.target, item.source),
            )
        }
        running_generator_loss = 0.0
        last_generator_snapshot = generator_baseline
        last_generator_snapshot_step = 0
        generator_pool_order = generator_training_pool[:]
        generator_rng.shuffle(generator_pool_order)
        generator_pool_index = 0
        for generator_step in range(1, args.generator_steps + 1):
            if generator_pool_index == len(generator_pool_order):
                generator_rng.shuffle(generator_pool_order)
                generator_pool_index = 0
            example = generator_pool_order[generator_pool_index]
            generator_pool_index += 1
            running_generator_loss += train_transformer_answer_generator_lesson(
                generator,
                generator_lessons[example],
                args.generator_learning_rate,
            )
            if (
                args.generator_eval_every > 0
                and generator_step % args.generator_eval_every == 0
            ):
                train_loss = running_generator_loss / args.generator_eval_every
                last_generator_snapshot = generator_snapshot(generator_step, train_loss)
                last_generator_snapshot_step = generator_step
                print(f"generator_step={generator_step} train_loss={train_loss:.4f}")
                running_generator_loss = 0.0

        if last_generator_snapshot_step != args.generator_steps:
            last_generator_snapshot = generator_snapshot(args.generator_steps, None)

        generator_checkpoint_path = args.run / "answer_generator.json"
        generator.save(generator_checkpoint_path)
        generator_metrics = {
            "architecture": "transformer-guided-answer-generator",
            "checkpoint": str(generator_checkpoint_path),
            "history": str(generator_history_path),
            "steps": args.generator_steps,
            "training_examples": len(generator_training_pool),
            "learning_rate": args.generator_learning_rate,
            "generator_eval_every": args.generator_eval_every,
            "max_answer_chars": args.generator_max_answer_chars,
            "transformer_top_k": args.generator_transformer_top_k,
            "labels": len(generator.config.labels),
            "features": len(generator.config.features),
            "baseline": generator_baseline,
            "final": last_generator_snapshot,
            "uses_answer_candidates": False,
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "training_data": "closed_world_lm.answer_model corpus-derived AnswerExample lessons",
        }

    checkpoint_path = args.run / "transformer_answer.json"
    model.save(checkpoint_path, tokenizer)
    tokenizer.save(args.run / "tokenizer.json")
    metrics = {
        "architecture": "tiny-decoder-only-transformer",
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "lessons": str(lessons_path),
        "steps": args.steps,
        "examples": len(examples),
        "training_examples": len(training_pool),
        "candidate_count": len(candidates),
        "training_candidate_count": len(training_candidates),
        "candidate_scope": args.candidate_scope,
        "include_completions": args.include_completions,
        "target_loss_weight": args.target_loss_weight,
        "choice_loss_weight": args.choice_loss_weight,
        "choice_negatives": args.choice_negatives,
        "choice_max_chars": args.choice_max_chars,
        "vocab_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "use_layer_norm": args.use_layer_norm,
        "layer_norm_epsilon": args.layer_norm_epsilon,
        "context_coverage": context_coverage,
        "baseline": baseline,
        "final": last_snapshot,
        "direct_answer": direct_answer_metrics,
        "answer_selector": selector_metrics,
        "answer_generator": generator_metrics,
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "tokenizer": "closed_world_lm.tokenizer.CharTokenizer",
        "training_data": "closed_world_lm.answer_model corpus-derived AnswerExample lessons",
    }
    with (args.run / "transformer_answer_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        train_transformer(args)
        return 0
    if args.command == "eval":
        result = eval_transformer(args)
        print(json.dumps(result["evals"], indent=2, sort_keys=True))
        return 0
    if args.command == "answer-train":
        train_transformer_answers(args)
        return 0
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
