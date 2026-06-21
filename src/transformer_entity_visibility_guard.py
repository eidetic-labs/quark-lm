"""Guard: the query entity must be inside the model's context window.

If ``context_size`` is too small for the prompt, ``make_context`` windows the
entity out: two prompts that differ only in the person (e.g. "where is mia's
ball?" vs "where is noah's ball?") produce a byte-identical context, so the
model literally cannot condition on the entity and entity-conditioned abstention
is impossible. This exposes a fast, model-free structural check used to lock that
diagnosis as a regression test.
"""

from __future__ import annotations

from typing import Any

from neural_char_ops import make_context


def windowed_context(tokenizer: Any, prompt: str, context_size: int) -> list[int]:
    """The token context the model actually sees for a prompt at this window."""

    return make_context(tokenizer.encode(prompt), context_size, tokenizer.pad_id)


def entity_visible_in_window(
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    context_size: int,
) -> bool:
    """True iff two entity-swapped prompts produce different windowed contexts.

    False means the differing entity has been windowed out — the conditioning
    variable is absent from the model input.
    """

    return windowed_context(tokenizer, prompt_a, context_size) != windowed_context(
        tokenizer, prompt_b, context_size
    )
