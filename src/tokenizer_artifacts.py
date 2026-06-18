"""Tokenizer proposal artifacts for closed-world subword updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from closed_world_subword_tokenizer import ClosedWorldSubwordTokenizer
from tokenizer import CharTokenizer
from tokenizer_scoring import TokenCandidate, score_subword_candidates


TOKENIZER_ARTIFACT_VERSION = 1


def propose_closed_world_subword_tokenizer(
    text: str,
    *,
    source_files: list[str] | None = None,
    protected_answers: set[str] | None = None,
    max_token_chars: int = 4,
    max_new_tokens: int = 32,
    base_tokenizer: CharTokenizer | ClosedWorldSubwordTokenizer | None = None,
) -> dict[str, Any]:
    tokenizer = _initial_tokenizer(text, base_tokenizer)
    accepted: list[TokenCandidate] = []
    rejected_by_token: dict[str, TokenCandidate] = {}

    for _ in range(max_new_tokens):
        candidates = score_subword_candidates(
            tokenizer,
            text,
            max_token_chars=max_token_chars,
            protected_answers=protected_answers,
        )
        for candidate in candidates:
            if not candidate.accepted:
                rejected_by_token.setdefault(candidate.token, candidate)
        best = next((candidate for candidate in candidates if candidate.accepted), None)
        if best is None:
            break
        tokenizer = tokenizer.with_merge(best.to_merge_rule())
        accepted.append(best)

    manifest = _manifest(
        text,
        tokenizer,
        accepted,
        list(rejected_by_token.values()),
        source_files or [],
        max_token_chars,
    )
    report = _report(text, tokenizer, accepted, protected_answers or set())
    return {
        "tokenizer": tokenizer,
        "manifest": manifest,
        "report": report,
        "manifest_hash": stable_json_hash(manifest),
    }


def write_tokenizer_artifacts(
    manifest_path: Path,
    report_path: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_json(manifest), encoding="utf-8")
    report_path.write_text(_json(report), encoding="utf-8")


def stable_json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _initial_tokenizer(
    text: str,
    base_tokenizer: CharTokenizer | ClosedWorldSubwordTokenizer | None,
) -> ClosedWorldSubwordTokenizer:
    if base_tokenizer is None:
        return ClosedWorldSubwordTokenizer.from_char_tokens(CharTokenizer.train(text).tokens)
    extended = base_tokenizer.extend(text)
    if isinstance(extended, ClosedWorldSubwordTokenizer):
        return extended
    return ClosedWorldSubwordTokenizer.from_char_tokens(extended.tokens)


def _manifest(
    text: str,
    tokenizer: ClosedWorldSubwordTokenizer,
    accepted: list[TokenCandidate],
    rejected: list[TokenCandidate],
    source_files: list[str],
    max_token_chars: int,
) -> dict[str, Any]:
    return {
        "version": TOKENIZER_ARTIFACT_VERSION,
        "tokenizer_type": tokenizer.tokenizer_type,
        "corpus_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_files": source_files,
        "max_token_chars": max_token_chars,
        "tokens": tokenizer.tokens,
        "merge_rules": [rule.__dict__ for rule in tokenizer.merge_rules],
        "candidate_scores": [candidate.to_dict() for candidate in accepted],
        "rejected_candidates": [candidate.to_dict() for candidate in rejected],
        "purity": {
            "pretrained_tokenizer": False,
            "external_vocabulary": False,
            "normalization": "none",
            "admitted_corpus_only": True,
        },
    }


def _report(
    text: str,
    tokenizer: ClosedWorldSubwordTokenizer,
    accepted: list[TokenCandidate],
    protected_answers: set[str],
) -> dict[str, Any]:
    char_tokenizer = CharTokenizer.train(text)
    char_count = len(char_tokenizer.encode(text))
    subword_count = len(tokenizer.encode(text))
    full_answer_tokens = [
        token
        for token in tokenizer.tokens
        if token in protected_answers or token.strip() in {answer.strip() for answer in protected_answers}
    ]
    return {
        "version": TOKENIZER_ARTIFACT_VERSION,
        "tokenizer_type": tokenizer.tokenizer_type,
        "round_trip_ok": tokenizer.decode(tokenizer.encode(text)) == text,
        "char_token_count": char_count,
        "subword_token_count": subword_count,
        "token_count_savings": char_count - subword_count,
        "compression_ratio": round(subword_count / max(char_count, 1), 6),
        "accepted_token_count": len(accepted),
        "max_target_token_chars": max((len(candidate.token) for candidate in accepted), default=1),
        "full_answer_tokens": full_answer_tokens,
        "branch_diversity_score": round(
            sum(candidate.context_diversity for candidate in accepted) / max(len(accepted), 1),
            6,
        ),
        "long_answer_effect": {
            "measured": False,
            "reason": "requires benchmark prompts and model output diagnostics",
        },
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
