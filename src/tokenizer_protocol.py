"""Shared tokenizer interface for closed-world model components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    tokens: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def tokenizer_type(self) -> str:
        raise NotImplementedError

    @property
    def pad_id(self) -> int:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def extend(self, text: str) -> "TokenizerProtocol":
        raise NotImplementedError

    def extends(self, base: "TokenizerProtocol") -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError
