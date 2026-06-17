"""Core model and math fixtures used by transformer tests."""

from __future__ import annotations

from answer_model import AnswerExample
from neural_char_metrics import continuation_nll
from neural_char_ops import context_before
from tokenizer import CharTokenizer
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_eval import score_transformer_records
from transformer_math import exclude_scalars, flatten_scalars, generation_distribution
from transformer_model import GenerationConfig, OptimizationConfig, TransformerConfig
from transformer_optimizer import ScalarOptimizer
from transformer_optimizer import load_optimizer_state, save_optimizer_state
from transformer_tiny_lm import TinyTransformerLM

__all__ = [
    "ANSWER_TERMINATOR",
    "AnswerExample",
    "CharTokenizer",
    "GenerationConfig",
    "OptimizationConfig",
    "ScalarOptimizer",
    "TinyTransformerLM",
    "TransformerConfig",
    "context_before",
    "continuation_nll",
    "exclude_scalars",
    "flatten_scalars",
    "generation_distribution",
    "load_optimizer_state",
    "save_optimizer_state",
    "score_transformer_records",
]
