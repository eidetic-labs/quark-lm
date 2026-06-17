"""Feature extraction helpers for the answer decoder."""

from __future__ import annotations

from answer_decoder_constants import BOS
from answer_model import feature_names


def decoder_feature_names(prompt: str, prefix: str) -> list[str]:
    names = feature_names(prompt)
    previous = prefix[-1] if prefix else BOS
    previous_two = prefix[-2:] if len(prefix) >= 2 else BOS
    names.extend(
        [
            f"pos:{len(prefix)}",
            f"prev:{previous}",
            f"prev2:{previous_two}",
            f"prefix:{prefix}",
        ]
    )
    return names
