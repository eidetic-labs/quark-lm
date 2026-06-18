"""Small closed-world benchmarks for transformer learning claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from answer_examples import AnswerExample
from neural_char_metrics import continuation_nll
from neural_char_ops import context_before
from probes import summarize
from tokenizer import CharTokenizer
from transformer_model import TransformerConfig


BENCHMARK_ID = "meaningful-micro-corpus-v2"


@dataclass(frozen=True)
class MicroBenchmarkConfig:
    context_size: int = 20
    embedding_dim: int = 4
    feedforward_dim: int = 8
    seed: int = 2
    epochs: int = 180
    learning_rate: float = 0.05


def meaningful_micro_corpus_examples() -> tuple[list[AnswerExample], list[AnswerExample]]:
    train = [
        AnswerExample("q place mia ball\nA:", " box.", "train:place"),
        AnswerExample("q color mia ball\nA:", " red.", "train:color"),
        AnswerExample("q place leo cube\nA:", " cup.", "train:place"),
        AnswerExample("q color leo cube\nA:", " tan.", "train:color"),
    ]
    heldout = [
        AnswerExample("place mia ball\nA:", " box.", "heldout:place"),
        AnswerExample("color mia ball\nA:", " red.", "heldout:color"),
        AnswerExample("place leo cube\nA:", " cup.", "heldout:place"),
        AnswerExample("color leo cube\nA:", " tan.", "heldout:color"),
    ]
    return train, heldout


def run_meaningful_micro_benchmark(
    model_cls: Any,
    config: MicroBenchmarkConfig | None = None,
) -> dict[str, Any]:
    config = config or MicroBenchmarkConfig()
    train_examples, heldout_examples = meaningful_micro_corpus_examples()
    tokenizer = CharTokenizer.train(_training_text(train_examples))
    model = model_cls.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=config.context_size,
            embedding_dim=config.embedding_dim,
            feedforward_dim=config.feedforward_dim,
            seed=config.seed,
        )
    )
    for _epoch in range(config.epochs):
        for example in train_examples:
            _train_example(model, tokenizer, example, config.learning_rate)

    candidates = sorted({example.target for example in train_examples})
    train_records = _score_examples(model, tokenizer, train_examples, candidates)
    heldout_records = _score_examples(model, tokenizer, heldout_examples, candidates)
    train_summary = summarize(train_records)
    heldout_summary = summarize(heldout_records)
    passed = (
        train_summary["exact_rate"] == 1.0
        and train_summary["candidate_rate"] == 1.0
        and heldout_summary["exact_rate"] == 1.0
        and heldout_summary["candidate_rate"] == 1.0
    )
    return {
        "benchmark_id": BENCHMARK_ID,
        "passed": passed,
        "config": asdict(config),
        "dataset": {
            "train_examples": len(train_examples),
            "heldout_examples": len(heldout_examples),
            "training_prompts": [example.prompt for example in train_examples],
            "heldout_prompts": [example.prompt for example in heldout_examples],
            "prompt_overlap": sorted(
                set(example.prompt for example in train_examples)
                & set(example.prompt for example in heldout_examples)
            ),
            "target_lengths": sorted({len(example.target) for example in train_examples}),
            "max_target_chars": max(len(example.target) for example in train_examples),
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
        },
        "train": {"summary": train_summary, "records": train_records},
        "heldout": {"summary": heldout_summary, "records": heldout_records},
    }


def _training_text(examples: list[AnswerExample]) -> str:
    return "".join(f"{example.prompt}{example.target}\n" for example in examples)


def _train_example(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    learning_rate: float,
) -> None:
    ids = tokenizer.encode(example.prompt + example.target)
    start = len(tokenizer.encode(example.prompt))
    for position in range(start, len(ids)):
        context = context_before(
            ids,
            position,
            model.config.context_size,
            tokenizer.pad_id,
        )
        model.train_step(context, ids[position], learning_rate)


def _score_examples(
    model: Any,
    tokenizer: CharTokenizer,
    examples: list[AnswerExample],
    candidates: list[str],
) -> list[dict[str, Any]]:
    records = []
    for index, example in enumerate(examples):
        scores = [
            {
                "target": candidate,
                "target_nll": continuation_nll(
                    model,
                    tokenizer,
                    example.prompt,
                    candidate,
                ),
            }
            for candidate in candidates
        ]
        predicted = min(scores, key=lambda item: float(item["target_nll"]))["target"]
        completion = model.generate(tokenizer, example.prompt, len(example.target))
        records.append(
            {
                "id": f"{example.source}:{index}",
                "prompt": example.prompt,
                "target": example.target,
                "completion": completion,
                "exact_match": completion == example.target,
                "candidate_match": predicted == example.target,
                "predicted_candidate": predicted,
                "candidate_scores": scores,
                "target_nll": continuation_nll(
                    model,
                    tokenizer,
                    example.prompt,
                    example.target,
                ),
            }
        )
    return records
