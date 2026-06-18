"""Append-only subword tokenizer trained only from admitted text."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tokenizer import PAD_TOKEN


SUBWORD_TOKENIZER_TYPE = "closed-world-subword"


@dataclass(frozen=True)
class MergeRule:
    left: str
    right: str
    token: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MergeRule":
        return cls(str(payload["left"]), str(payload["right"]), str(payload["token"]))


@dataclass
class ClosedWorldSubwordTokenizer:
    tokens: list[str]
    merge_rules: list[MergeRule]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != PAD_TOKEN:
            raise ValueError(f"first token must be {PAD_TOKEN!r}")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("token vocabulary cannot contain duplicates")
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for index, token in enumerate(self.tokens)}
        self._validate_merge_rules()

    @property
    def tokenizer_type(self) -> str:
        return SUBWORD_TOKENIZER_TYPE

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_char_tokens(cls, tokens: list[str]) -> "ClosedWorldSubwordTokenizer":
        return cls(list(tokens), [])

    def with_merge(self, rule: MergeRule) -> "ClosedWorldSubwordTokenizer":
        if rule.token in self.stoi:
            raise ValueError(f"token {rule.token!r} already exists")
        if rule.left not in self.stoi or rule.right not in self.stoi:
            raise ValueError("merge rule parts must already exist")
        return ClosedWorldSubwordTokenizer(
            [*self.tokens, rule.token],
            [*self.merge_rules, rule],
        )

    def extend(self, text: str) -> "ClosedWorldSubwordTokenizer":
        chars = sorted(set(text))
        if PAD_TOKEN in chars:
            raise ValueError(f"training text cannot contain reserved token {PAD_TOKEN!r}")
        additions = [char for char in chars if char not in self.stoi]
        return ClosedWorldSubwordTokenizer(
            [*self.tokens, *additions],
            list(self.merge_rules),
        )

    def extends(self, base: Any) -> bool:
        base_tokens = getattr(base, "tokens", [])
        if self.tokens[: len(base_tokens)] != list(base_tokens):
            return False
        base_rules = getattr(base, "merge_rules", [])
        return self.merge_rules[: len(base_rules)] == list(base_rules)

    def encode(self, text: str) -> list[int]:
        pieces = list(text)
        for piece in pieces:
            if piece not in self.stoi:
                raise ValueError(f"character {piece!r} is outside the admitted vocabulary")
        for rule in self.merge_rules:
            pieces = _apply_rule(pieces, rule)
        return [self.stoi[piece] for piece in pieces]

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            token = self.itos[token_id]
            if token != PAD_TOKEN:
                pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "tokens": self.tokens,
            "merge_rules": [asdict(rule) for rule in self.merge_rules],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClosedWorldSubwordTokenizer":
        return cls(
            list(payload["tokens"]),
            [MergeRule.from_dict(rule) for rule in payload.get("merge_rules", [])],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "ClosedWorldSubwordTokenizer":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def _validate_merge_rules(self) -> None:
        seen = set(self.tokens[:1])
        for token in self.tokens[1:]:
            seen.add(token)
        for rule in self.merge_rules:
            if rule.left not in self.stoi or rule.right not in self.stoi:
                raise ValueError("merge rule references unknown token")
            if rule.token not in self.stoi:
                raise ValueError("merge rule output must exist in vocabulary")


def _apply_rule(pieces: list[str], rule: MergeRule) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(pieces):
        if (
            index + 1 < len(pieces)
            and pieces[index] == rule.left
            and pieces[index + 1] == rule.right
        ):
            output.append(rule.token)
            index += 2
        else:
            output.append(pieces[index])
            index += 1
    return output
