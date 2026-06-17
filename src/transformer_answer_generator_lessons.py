"""Cached lessons for transformer-guided answer-generator training."""

from __future__ import annotations

import math
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_answer_generator_constants import GENERATOR_EOS
from transformer_answer_generator_model import TransformerGuidedAnswerGenerator
from transformer_math import softmax_floats


GeneratorLesson = list[tuple[int, dict[int, float]]]


def transformer_answer_generator_lesson(
    generator: TransformerGuidedAnswerGenerator,
    model: Any,
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
        generator._apply_gradients(probs, features, learning_rate)
    return total / max(len(lesson), 1)
