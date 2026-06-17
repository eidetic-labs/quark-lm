"""Build answer decoder instances from closed-world examples."""

from __future__ import annotations

from answer_decoder_constants import EOS
from answer_decoder_features import decoder_feature_names
from answer_decoder_model import AnswerDecoder, AnswerDecoderConfig
from answer_model import AnswerExample


def build_decoder(
    examples: list[AnswerExample],
    seed: int,
    max_answer_chars: int,
) -> AnswerDecoder:
    labels = sorted({char for example in examples for char in example.target} | {EOS})
    feature_set = set[str]()
    for example in examples:
        prefix = ""
        for label in [*example.target, EOS]:
            feature_set.update(decoder_feature_names(example.prompt, prefix))
            if label != EOS:
                prefix += label
    config = AnswerDecoderConfig(
        labels=labels,
        features=sorted(feature_set),
        seed=seed,
        max_answer_chars=max_answer_chars,
    )
    return AnswerDecoder.init_random(config)
