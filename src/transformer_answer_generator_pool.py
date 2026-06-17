"""Training-pool weighting for transformer-guided answer generators."""

from __future__ import annotations

from answer_model import AnswerExample


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
