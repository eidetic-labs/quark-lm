"""Compatibility facade for the closed-world answer model."""

from __future__ import annotations

from answer_examples import (
    AnswerExample,
    examples_from_sources,
    glossary_prompt_templates,
    learning_prompt_templates,
    prompt_templates,
    self_prompt_templates,
)
from answer_features import (
    GLOSSARY_PROMPT_PATTERNS,
    SELF_PROMPT_PATTERNS,
    SEMANTIC_FEATURE_WEIGHT,
    SEMANTIC_PROMPT_PATTERNS,
    WORD_RE,
    feature_names,
    semantic_feature_names,
)
from answer_model_builder import build_model
from answer_model_cli import main, parse_args
from answer_model_commands import eval_model
from answer_model_constants import (
    DEFAULT_EVALS,
    DEFAULT_RUN_DIR,
    DEFAULT_TRAIN_TEXT,
    PROJECT_DIR,
)
from answer_model_data import load_training_examples, write_lessons
from answer_model_evaluation import evaluate_records, summarize_eval
from answer_model_pool import answer_training_pool
from answer_model_softmax import AnswerModelConfig, AnswerSoftmax, softmax
from answer_model_training import train_model
from curriculum import DEFAULT_CORPUS_DIR, DEFAULT_OUTPUT_DIR, read_json
from probes import read_jsonl


__all__ = [
    "DEFAULT_CORPUS_DIR",
    "DEFAULT_EVALS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RUN_DIR",
    "DEFAULT_TRAIN_TEXT",
    "GLOSSARY_PROMPT_PATTERNS",
    "PROJECT_DIR",
    "SELF_PROMPT_PATTERNS",
    "SEMANTIC_FEATURE_WEIGHT",
    "SEMANTIC_PROMPT_PATTERNS",
    "WORD_RE",
    "AnswerExample",
    "AnswerModelConfig",
    "AnswerSoftmax",
    "answer_training_pool",
    "build_model",
    "eval_model",
    "evaluate_records",
    "examples_from_sources",
    "feature_names",
    "glossary_prompt_templates",
    "learning_prompt_templates",
    "load_training_examples",
    "main",
    "parse_args",
    "prompt_templates",
    "read_json",
    "read_jsonl",
    "self_prompt_templates",
    "semantic_feature_names",
    "softmax",
    "summarize_eval",
    "train_model",
    "write_lessons",
]


if __name__ == "__main__":
    raise SystemExit(main())
