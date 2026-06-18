"""A character tokenizer trained only from admitted text."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAD_TOKEN = "<pad>"
CHAR_TOKENIZER_TYPE = "char"


@dataclass
class CharTokenizer:
    tokens: list[str]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != PAD_TOKEN:
            raise ValueError(f"first token must be {PAD_TOKEN!r}")
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def tokenizer_type(self) -> str:
        return CHAR_TOKENIZER_TYPE

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        if PAD_TOKEN in chars:
            raise ValueError(f"training text cannot contain reserved token {PAD_TOKEN!r}")
        return cls([PAD_TOKEN, *chars])

    def extend(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        if PAD_TOKEN in chars:
            raise ValueError(f"training text cannot contain reserved token {PAD_TOKEN!r}")
        additions = [char for char in chars if char not in self.stoi]
        if not additions:
            return CharTokenizer(list(self.tokens))
        return CharTokenizer([*self.tokens, *additions])

    def extends(self, base: "CharTokenizer") -> bool:
        return self.tokens[: base.vocab_size] == base.tokens

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for char in text:
            try:
                ids.append(self.stoi[char])
            except KeyError as exc:
                raise ValueError(f"character {char!r} is outside the admitted vocabulary") from exc
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            token = self.itos[token_id]
            if token != PAD_TOKEN:
                pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict[str, Any]:
        return {"tokenizer_type": self.tokenizer_type, "tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharTokenizer":
        return cls(list(payload["tokens"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
