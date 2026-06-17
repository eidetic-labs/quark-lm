"""Feature extraction for transformer-guided answer generation."""

from __future__ import annotations

from typing import Any

from answer_model import feature_names
from neural_char_model import make_context
from tokenizer import CharTokenizer
from transformer_answer_generator_constants import GENERATOR_BOS


def transformer_answer_generator_feature_names(
    model: Any,
    tokenizer: CharTokenizer,
    prompt: str,
    prefix: str,
    transformer_top_k: int,
) -> list[str]:
    names = feature_names(prompt)
    previous = prefix[-1] if prefix else GENERATOR_BOS
    previous_two = prefix[-2:] if len(prefix) >= 2 else GENERATOR_BOS
    names.extend(
        [
            f"pos:{len(prefix)}",
            f"prev:{previous}",
            f"prev2:{previous_two}",
            f"prefix:{prefix}",
        ]
    )
    context_ids = tokenizer.encode(prompt + prefix)
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    probs = model.predict(context)
    top_count = max(0, min(transformer_top_k, len(probs)))
    top_ids = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)[
        :top_count
    ]
    for rank, token_id in enumerate(top_ids):
        token = tokenizer.itos[token_id]
        names.append(f"transformer_top:{rank}:{token!r}")
        if rank == 0:
            names.append(f"transformer_argmax:{token!r}")
    return names
