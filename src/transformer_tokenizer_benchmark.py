"""Tokenizer comparison benchmarks for transformer learning diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from answer_examples import AnswerExample
from tokenizer import CharTokenizer
from tokenizer_artifacts import propose_closed_world_subword_tokenizer
from tokenizer_artifact_validation import validate_tokenizer_artifacts


BENCHMARK_ID = "tokenizer-comparison-v1"


@dataclass(frozen=True)
class TokenizerBenchmarkConfig:
    max_token_chars: int = 4
    max_new_tokens: int = 16


def tokenizer_benchmark_examples() -> tuple[list[AnswerExample], list[AnswerExample]]:
    short = [
        AnswerExample("q place mia ball\nA:", " box.", "short"),
        AnswerExample("q color mia ball\nA:", " red.", "short"),
    ]
    long = [
        AnswerExample("q place nia ship\nA:", " kitchen.", "long"),
        AnswerExample("q thing nia shape\nA:", " kite.", "long"),
        AnswerExample("q place ria cup\nA:", " kitchen shelf.", "long"),
    ]
    return short, long


def run_tokenizer_comparison_benchmark(
    config: TokenizerBenchmarkConfig | None = None,
) -> dict[str, Any]:
    config = config or TokenizerBenchmarkConfig()
    short_examples, long_examples = tokenizer_benchmark_examples()
    examples = [*short_examples, *long_examples]
    text = "".join(f"{example.prompt}{example.target}\n" for example in examples)
    protected_answers = {example.target for example in examples}
    char_tokenizer = CharTokenizer.train(text)
    proposal = propose_closed_world_subword_tokenizer(
        text,
        protected_answers=protected_answers,
        max_token_chars=config.max_token_chars,
        max_new_tokens=config.max_new_tokens,
    )
    validate_tokenizer_artifacts(
        proposal["manifest"],
        proposal["report"],
        manifest_hash=proposal["manifest_hash"],
    )
    subword_tokenizer = proposal["tokenizer"]
    records = [
        _score_example(char_tokenizer, subword_tokenizer, example)
        for example in examples
    ]
    long_records = [record for record in records if record["source"] == "long"]
    full_answer_tokens = proposal["report"]["full_answer_tokens"]
    passed = (
        subword_tokenizer.decode(subword_tokenizer.encode(text)) == text
        and not full_answer_tokens
        and sum(record["subword_target_tokens"] for record in long_records)
        < sum(record["char_target_tokens"] for record in long_records)
    )
    return {
        "benchmark_id": BENCHMARK_ID,
        "passed": passed,
        "config": asdict(config),
        "tokenizer_manifest_hash": proposal["manifest_hash"],
        "manifest": proposal["manifest"],
        "report": proposal["report"],
        "records": records,
    }


def _score_example(
    char_tokenizer: CharTokenizer,
    subword_tokenizer: Any,
    example: AnswerExample,
) -> dict[str, Any]:
    char_prompt = len(char_tokenizer.encode(example.prompt))
    char_target = len(char_tokenizer.encode(example.target))
    subword_prompt = len(subword_tokenizer.encode(example.prompt))
    subword_target = len(subword_tokenizer.encode(example.target))
    return {
        "source": example.source,
        "prompt": example.prompt,
        "target": example.target,
        "char_prompt_tokens": char_prompt,
        "char_target_tokens": char_target,
        "subword_prompt_tokens": subword_prompt,
        "subword_target_tokens": subword_target,
        "target_token_savings": char_target - subword_target,
    }
