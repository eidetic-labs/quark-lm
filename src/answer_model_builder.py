"""Construction helpers for answer model checkpoints."""

from __future__ import annotations

from answer_examples import AnswerExample
from answer_features import feature_names
from answer_model_softmax import AnswerModelConfig, AnswerSoftmax


def build_model(examples: list[AnswerExample], seed: int) -> AnswerSoftmax:
    labels = sorted({example.target for example in examples})
    feature_set = set[str]()
    for example in examples:
        feature_set.update(feature_names(example.prompt))
    config = AnswerModelConfig(labels=labels, features=sorted(feature_set), seed=seed)
    return AnswerSoftmax.init_random(config)
