"""Compatibility facade for the closed-world answer decoder."""

from __future__ import annotations

from answer_decoder_artifacts import write_lessons
from answer_decoder_builder import build_decoder
from answer_decoder_cli import main, parse_args
from answer_decoder_commands import eval_decoder
from answer_decoder_constants import (
    BOS,
    DECODER_GLOSSARY_REPEATS,
    DECODER_SELF_LEARNING_REPEATS,
    DEFAULT_DECODER_RUN_DIR,
    EOS,
)
from answer_decoder_evaluation import evaluate_records, summarize_eval
from answer_decoder_features import decoder_feature_names
from answer_decoder_model import AnswerDecoder, AnswerDecoderConfig, softmax
from answer_decoder_pool import decoder_training_pool
from answer_decoder_training import train_decoder
from answer_model import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVALS,
    DEFAULT_RUN_DIR,
    DEFAULT_TRAIN_TEXT,
    AnswerExample,
    examples_from_sources,
    feature_names,
    load_training_examples,
)


__all__ = [
    "BOS",
    "DECODER_GLOSSARY_REPEATS",
    "DECODER_SELF_LEARNING_REPEATS",
    "DEFAULT_DECODER_RUN_DIR",
    "DEFAULT_CORPUS_DIR",
    "DEFAULT_EVALS",
    "DEFAULT_RUN_DIR",
    "DEFAULT_TRAIN_TEXT",
    "EOS",
    "AnswerExample",
    "AnswerDecoder",
    "AnswerDecoderConfig",
    "build_decoder",
    "decoder_feature_names",
    "decoder_training_pool",
    "eval_decoder",
    "evaluate_records",
    "examples_from_sources",
    "feature_names",
    "load_training_examples",
    "main",
    "parse_args",
    "softmax",
    "summarize_eval",
    "train_decoder",
    "write_lessons",
]


if __name__ == "__main__":
    raise SystemExit(main())
