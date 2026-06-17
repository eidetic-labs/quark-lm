"""Training-pool scheduling for answer decoder examples."""

from __future__ import annotations

from answer_decoder_constants import (
    DECODER_GLOSSARY_REPEATS,
    DECODER_SELF_LEARNING_REPEATS,
)
from answer_model import AnswerExample


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
