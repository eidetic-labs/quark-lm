"""A generative closed-world answer decoder trained from admitted lessons."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .answer_model import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVALS,
    DEFAULT_RUN_DIR,
    DEFAULT_TRAIN_TEXT,
    AnswerExample,
    examples_from_sources,
    feature_names,
    load_training_examples,
)
from .probes import read_jsonl


EOS = "<eos>"
BOS = "<bos>"
DEFAULT_DECODER_RUN_DIR = DEFAULT_RUN_DIR.parent / "answer-decoder-latest"
DECODER_SELF_LEARNING_REPEATS = 55
DECODER_GLOSSARY_REPEATS = 24


@dataclass
class AnswerDecoderConfig:
    labels: list[str]
    features: list[str]
    seed: int = 7
    max_answer_chars: int = 32


class AnswerDecoder:
    def __init__(
        self,
        config: AnswerDecoderConfig,
        weights: list[list[float]],
        bias: list[float],
    ) -> None:
        self.config = config
        self.weights = weights
        self.bias = bias
        self.label_to_index = {label: index for index, label in enumerate(config.labels)}
        self.feature_to_index = {feature: index for index, feature in enumerate(config.features)}

    @classmethod
    def init_random(cls, config: AnswerDecoderConfig) -> "AnswerDecoder":
        rng = random.Random(config.seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in config.features]
            for _ in config.labels
        ]
        bias = [0.0 for _ in config.labels]
        return cls(config, weights, bias)

    def featurize(self, prompt: str, prefix: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for name in decoder_feature_names(prompt, prefix):
            index = self.feature_to_index.get(name)
            if index is not None:
                counts[index] = counts.get(index, 0.0) + 1.0
        return counts

    def probabilities(self, prompt: str, prefix: str) -> list[float]:
        return softmax(self._logits(self.featurize(prompt, prefix)))

    def predict_next(self, prompt: str, prefix: str) -> str:
        probs = self.probabilities(prompt, prefix)
        index = max(range(len(probs)), key=lambda item: probs[item])
        return self.config.labels[index]

    def generate(self, prompt: str) -> str:
        prefix = ""
        for _ in range(self.config.max_answer_chars):
            label = self.predict_next(prompt, prefix)
            if label == EOS:
                break
            prefix += label
        return prefix

    def sequence_loss(self, prompt: str, target: str) -> float:
        prefix = ""
        total = 0.0
        labels = [*target, EOS]
        for label in labels:
            probs = self.probabilities(prompt, prefix)
            total += -math.log(max(probs[self.label_to_index[label]], 1e-12))
            if label != EOS:
                prefix += label
        return total / len(labels)

    def train_example(self, example: AnswerExample, learning_rate: float) -> float:
        prefix = ""
        total = 0.0
        labels = [*example.target, EOS]
        for label in labels:
            target_index = self.label_to_index[label]
            features = self.featurize(example.prompt, prefix)
            probs = softmax(self._logits(features))
            total += -math.log(max(probs[target_index], 1e-12))
            probs[target_index] -= 1.0
            for label_index, grad in enumerate(probs):
                self.bias[label_index] -= learning_rate * grad
                for feature_index, value in features.items():
                    self.weights[label_index][feature_index] -= learning_rate * grad * value
            if label != EOS:
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
            "config": asdict(self.config),
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerDecoder":
        return cls(
            AnswerDecoderConfig(**payload["config"]),
            payload["weights"],
            payload["bias"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "AnswerDecoder":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def decoder_feature_names(prompt: str, prefix: str) -> list[str]:
    names = feature_names(prompt)
    previous = prefix[-1] if prefix else BOS
    previous_two = prefix[-2:] if len(prefix) >= 2 else BOS
    names.extend(
        [
            f"pos:{len(prefix)}",
            f"prev:{previous}",
            f"prev2:{previous_two}",
            f"prefix:{prefix}",
        ]
    )
    return names


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(item - max_logit) for item in logits]
    total = sum(exps)
    return [item / total for item in exps]


def build_decoder(examples: list[AnswerExample], seed: int, max_answer_chars: int) -> AnswerDecoder:
    labels = sorted({char for example in examples for char in example.target} | {EOS})
    feature_set = set[str]()
    for example in examples:
        prefix = ""
        for label in [*example.target, EOS]:
            feature_set.update(decoder_feature_names(example.prompt, prefix))
            if label != EOS:
                prefix += label
    config = AnswerDecoderConfig(
        labels=labels,
        features=sorted(feature_set),
        seed=seed,
        max_answer_chars=max_answer_chars,
    )
    return AnswerDecoder.init_random(config)


def decoder_training_pool(examples: list[AnswerExample]) -> list[AnswerExample]:
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
            repeats += DECODER_SELF_LEARNING_REPEATS
        if example.source.endswith(":glossary"):
            repeats += DECODER_GLOSSARY_REPEATS
        pool.extend([example] * repeats)
    return pool


def evaluate_records(model: AnswerDecoder, records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    total_loss = 0.0
    for record in records:
        prediction = model.generate(record["prompt"])
        loss = model.sequence_loss(record["prompt"], record["target"])
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "prediction": prediction,
                "exact_match": prediction == record["target"],
                "target_loss": loss,
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "records": scored,
    }


def summarize_eval(model: AnswerDecoder, records: list[dict[str, Any]]) -> dict[str, Any]:
    result = evaluate_records(model, records)
    failed_records = [record for record in result["records"] if not record["exact_match"]]
    return {
        "count": result["count"],
        "exact": result["exact"],
        "exact_rate": result["exact_rate"],
        "avg_target_loss": result["avg_target_loss"],
        "failed_records": failed_records,
    }


def write_lessons(examples: list[AnswerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")


def train_decoder(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_training_examples(args.train_text, args.corpus_dir)
    training_pool = decoder_training_pool(examples)
    model = build_decoder(examples, args.seed, args.max_answer_chars)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    history_path = args.run / "decoder_metrics.jsonl"
    lessons_path = args.run / "decoder_lessons.jsonl"
    write_lessons(examples, lessons_path)

    def snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
        result = {
            "step": step,
            "train_loss": train_loss,
            "evals": {
                path.stem: summarize_eval(model, read_jsonl(path))
                for path in DEFAULT_EVALS
            },
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        return result

    baseline = snapshot(0, None)
    running_loss = 0.0
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
        running_loss += model.train_example(example, args.learning_rate)
        if step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            last_snapshot = snapshot(step, train_loss)
            last_snapshot_step = step
            print(f"step={step} train_loss={train_loss:.4f}")
            running_loss = 0.0

    if last_snapshot_step != args.steps:
        last_snapshot = snapshot(args.steps, None)

    checkpoint_path = args.run / "answer_decoder.json"
    model.save(checkpoint_path)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "lessons": str(lessons_path),
        "steps": args.steps,
        "examples": len(examples),
        "training_examples": len(training_pool),
        "labels": len(model.config.labels),
        "features": len(model.config.features),
        "baseline": baseline,
        "final": last_snapshot,
    }
    with (args.run / "decoder_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def eval_decoder(args: argparse.Namespace) -> dict[str, Any]:
    model = AnswerDecoder.load(args.checkpoint)
    result = {
        path.stem: evaluate_records(model, read_jsonl(path))
        for path in DEFAULT_EVALS
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    summary = {
        name: {
            "count": value["count"],
            "exact": value["exact"],
            "exact_rate": value["exact_rate"],
            "avg_target_loss": value["avg_target_loss"],
        }
        for name, value in result.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    train.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    train.add_argument("--run", type=Path, default=DEFAULT_DECODER_RUN_DIR)
    train.add_argument("--steps", type=int, default=2200)
    train.add_argument("--learning-rate", type=float, default=0.04)
    train.add_argument("--eval-every", type=int, default=550)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--max-answer-chars", type=int, default=64)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_DECODER_RUN_DIR / "answer_decoder.json")
    evaluate.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        train_decoder(args)
        return 0
    if args.command == "eval":
        eval_decoder(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
