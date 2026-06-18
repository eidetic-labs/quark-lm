"""Tokenizer serialization helpers."""

from __future__ import annotations

from typing import Any

from closed_world_subword_tokenizer import (
    SUBWORD_TOKENIZER_TYPE,
    ClosedWorldSubwordTokenizer,
)
from tokenizer import CHAR_TOKENIZER_TYPE, CharTokenizer
from tokenizer_protocol import TokenizerProtocol


def tokenizer_from_dict(payload: dict[str, Any]) -> TokenizerProtocol:
    tokenizer_type = payload.get("tokenizer_type") or payload.get("type")
    if tokenizer_type in {None, CHAR_TOKENIZER_TYPE, "tokenizer.CharTokenizer"}:
        if "merge_rules" in payload:
            return ClosedWorldSubwordTokenizer.from_dict(payload)
        return CharTokenizer.from_dict(payload)
    if tokenizer_type == SUBWORD_TOKENIZER_TYPE:
        return ClosedWorldSubwordTokenizer.from_dict(payload)
    raise ValueError(f"unsupported tokenizer type: {tokenizer_type!r}")


def tokenizer_identity(tokenizer: Any) -> str:
    tokenizer_type = getattr(tokenizer, "tokenizer_type", CHAR_TOKENIZER_TYPE)
    if tokenizer_type == CHAR_TOKENIZER_TYPE:
        return "tokenizer.CharTokenizer"
    if tokenizer_type == SUBWORD_TOKENIZER_TYPE:
        return "closed_world_subword_tokenizer.ClosedWorldSubwordTokenizer"
    return str(tokenizer_type)
