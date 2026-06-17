"""Answer-training pool and configuration helpers."""

from __future__ import annotations

from answer_model import AnswerExample


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
