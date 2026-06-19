"""Closed-world boundary payloads for PyTorch training attempts."""

from __future__ import annotations

from typing import Any


def build_torch_training_attempt_boundary() -> dict[str, Any]:
    """Return the fixed boundary proof for optional PyTorch training attempts."""

    return {
        "runtime_library_allowed": True,
        "training_text_source": "admitted_curriculum",
        "learned_assets_imported": False,
        "training_data_imported": False,
        "pretrained_weights_imported": False,
        "pretrained_tokenizer_imported": False,
        "external_embeddings_imported": False,
    }


def torch_training_attempt_boundary_failures(
    boundary: dict[str, Any],
) -> list[str]:
    """Return closed-world boundary fields that diverge from the expected proof."""

    expected = build_torch_training_attempt_boundary()
    return [
        key
        for key, expected_value in expected.items()
        if not _matches_expected_boundary_value(
            actual=boundary.get(key),
            expected=expected_value,
        )
    ]


def _matches_expected_boundary_value(*, actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    return actual == expected
