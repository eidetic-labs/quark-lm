"""Guarded subword-candidate scoring."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from closed_world_subword_tokenizer import (
    ClosedWorldSubwordTokenizer,
    MergeRule,
)


BOUNDARY_LEFT = "<BOS>"
BOUNDARY_RIGHT = "<EOS>"


@dataclass(frozen=True)
class TokenCandidate:
    token: str
    left: str
    right: str
    frequency: int
    left_frequency: int
    right_frequency: int
    source_count: int
    context_diversity: int
    score: float
    rejection_reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["accepted"] = self.accepted
        return payload

    def to_merge_rule(self) -> MergeRule:
        return MergeRule(self.left, self.right, self.token)


def score_subword_candidates(
    tokenizer: ClosedWorldSubwordTokenizer,
    text: str,
    *,
    max_token_chars: int,
    protected_answers: set[str] | None = None,
) -> list[TokenCandidate]:
    protected = protected_answers or set()
    sequences = _token_sequences(tokenizer, text)
    word_counts = _word_counts(text)
    pair_counts: dict[tuple[str, str], int] = {}
    token_counts: dict[str, int] = {}
    left_contexts: dict[tuple[str, str], set[str]] = {}
    right_contexts: dict[tuple[str, str], set[str]] = {}
    source_ids: dict[tuple[str, str], set[int]] = {}

    for source_id, sequence in enumerate(sequences):
        for token in sequence:
            token_counts[token] = token_counts.get(token, 0) + 1
        for index in range(len(sequence) - 1):
            pair = (sequence[index], sequence[index + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            left_contexts.setdefault(pair, set()).add(
                sequence[index - 1] if index > 0 else BOUNDARY_LEFT
            )
            right_contexts.setdefault(pair, set()).add(
                sequence[index + 2] if index + 2 < len(sequence) else BOUNDARY_RIGHT
            )
            source_ids.setdefault(pair, set()).add(source_id)

    candidates = [
        _candidate_from_pair(
            tokenizer,
            left,
            right,
            frequency,
            token_counts,
            left_contexts[(left, right)],
            right_contexts[(left, right)],
            source_ids[(left, right)],
            max_token_chars,
            protected,
            word_counts,
        )
        for (left, right), frequency in pair_counts.items()
    ]
    return sorted(
        candidates,
        key=lambda item: (item.accepted, item.score, item.frequency, item.token),
        reverse=True,
    )


def _candidate_from_pair(
    tokenizer: ClosedWorldSubwordTokenizer,
    left: str,
    right: str,
    frequency: int,
    token_counts: dict[str, int],
    left_contexts: set[str],
    right_contexts: set[str],
    source_ids: set[int],
    max_token_chars: int,
    protected_answers: set[str],
    word_counts: dict[str, int],
) -> TokenCandidate:
    token = left + right
    context_diversity = len(left_contexts) + len(right_contexts)
    rejection_reasons = _rejection_reasons(
        tokenizer,
        token,
        max_token_chars,
        protected_answers,
        word_counts,
    )
    left_frequency = token_counts.get(left, 1)
    right_frequency = token_counts.get(right, 1)
    informativeness = frequency / max(left_frequency * right_frequency, 1)
    single_context_penalty = 1.0 if context_diversity <= 2 else 0.0
    score = (
        frequency * 2.0
        + informativeness * 10.0
        + context_diversity
        + len(source_ids)
        - single_context_penalty
        - len(token) * 0.05
    )
    return TokenCandidate(
        token=token,
        left=left,
        right=right,
        frequency=frequency,
        left_frequency=left_frequency,
        right_frequency=right_frequency,
        source_count=len(source_ids),
        context_diversity=context_diversity,
        score=round(score, 6),
        rejection_reasons=tuple(rejection_reasons),
    )


def _rejection_reasons(
    tokenizer: ClosedWorldSubwordTokenizer,
    token: str,
    max_token_chars: int,
    protected_answers: set[str],
    word_counts: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    if token in tokenizer.stoi:
        reasons.append("existing_token")
    if "\n" in token:
        reasons.append("newline_crossing")
    if len(token) > max_token_chars:
        reasons.append("excessive_length")
    if token in protected_answers or token.strip() in {answer.strip() for answer in protected_answers}:
        reasons.append("full_answer_token")
    if re.fullmatch(r"[A-Za-z]+", token) and word_counts.get(token.lower(), 0) == 1:
        reasons.append("whole_rare_word")
    return reasons


def _token_sequences(tokenizer: ClosedWorldSubwordTokenizer, text: str) -> list[list[str]]:
    sequences = []
    for line in text.splitlines():
        if not line:
            continue
        sequences.append([tokenizer.itos[token_id] for token_id in tokenizer.encode(line)])
    return sequences


def _word_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z]+", text.lower()):
        counts[word] = counts.get(word, 0) + 1
    return counts
