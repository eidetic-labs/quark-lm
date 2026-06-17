"""Configuration for transformer-guided answer generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerAnswerGeneratorConfig:
    labels: list[str]
    features: list[str]
    seed: int = 17
    max_answer_chars: int = 64
    transformer_top_k: int = 3
