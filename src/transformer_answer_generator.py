"""Compatibility facade for transformer-guided answer generation."""

from __future__ import annotations

from transformer_answer_generator_builder import build_transformer_answer_generator
from transformer_answer_generator_config import TransformerAnswerGeneratorConfig
from transformer_answer_generator_constants import GENERATOR_BOS, GENERATOR_EOS
from transformer_answer_generator_evaluation import evaluate_answer_generator_records
from transformer_answer_generator_features import transformer_answer_generator_feature_names
from transformer_answer_generator_lessons import (
    GeneratorLesson,
    train_transformer_answer_generator_lesson,
    transformer_answer_generator_lesson,
)
from transformer_answer_generator_model import TransformerGuidedAnswerGenerator
from transformer_answer_generator_pool import transformer_answer_generator_training_pool


__all__ = [
    "GENERATOR_BOS",
    "GENERATOR_EOS",
    "GeneratorLesson",
    "TransformerAnswerGeneratorConfig",
    "TransformerGuidedAnswerGenerator",
    "build_transformer_answer_generator",
    "evaluate_answer_generator_records",
    "train_transformer_answer_generator_lesson",
    "transformer_answer_generator_feature_names",
    "transformer_answer_generator_lesson",
    "transformer_answer_generator_training_pool",
]
