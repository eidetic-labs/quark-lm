from __future__ import annotations

from support.core import CharTokenizer, TinyTransformerLM, TransformerConfig, context_before

DEFAULT_TEXT = "abc abc\n"


def transformer_config(
    tokenizer: CharTokenizer,
    **overrides: object,
) -> TransformerConfig:
    values = {
        "vocab_size": tokenizer.vocab_size,
        "context_size": 4,
        "embedding_dim": 4,
        "feedforward_dim": 8,
        "seed": 1,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def char_model_fixture(
    text: str = DEFAULT_TEXT,
    **config_overrides: object,
) -> tuple[CharTokenizer, list[int], TransformerConfig, TinyTransformerLM]:
    tokenizer = CharTokenizer.train(text)
    ids = tokenizer.encode(text)
    config = transformer_config(tokenizer, **config_overrides)
    model = TinyTransformerLM.init_random(config)
    return tokenizer, ids, config, model


def context_and_target(
    ids: list[int],
    config: TransformerConfig,
    tokenizer: CharTokenizer,
    index: int = 4,
) -> tuple[list[int], int]:
    return context_before(ids, index, config.context_size, tokenizer.pad_id), ids[index]
