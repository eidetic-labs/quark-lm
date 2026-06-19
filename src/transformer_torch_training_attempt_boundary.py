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
