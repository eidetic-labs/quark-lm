"""Construction helpers for transformer-guided answer generators."""

from __future__ import annotations

from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_answer_generator_config import TransformerAnswerGeneratorConfig
from transformer_answer_generator_constants import GENERATOR_EOS
from transformer_answer_generator_features import (
    transformer_answer_generator_feature_names,
)
from transformer_answer_generator_model import TransformerGuidedAnswerGenerator


def build_transformer_answer_generator(
    examples: list[AnswerExample],
    model: Any,
    tokenizer: CharTokenizer,
    seed: int,
    max_answer_chars: int,
    transformer_top_k: int,
) -> TransformerGuidedAnswerGenerator:
    labels = sorted(
        {char for example in examples for char in example.target} | {GENERATOR_EOS}
    )
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
