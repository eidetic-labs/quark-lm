"""Deterministic closed-world memory index."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from curriculum import DEFAULT_CORPUS_DIR
from memory_cards import MemoryCard, build_memory_cards
from memory_retrieval_signatures import (
    prompt_signature,
    signatures_match,
    tokenize,
)


UNKNOWN_ANSWER = " unknown."


class ClosedWorldMemoryIndex:
    def __init__(self, cards: list[MemoryCard]) -> None:
        self.cards = list(cards)

    @classmethod
    def from_corpus(cls, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> "ClosedWorldMemoryIndex":
        return cls(build_memory_cards(corpus_dir))

    def retrieve(self, prompt: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_signature = prompt_signature(prompt)
        candidates = (
            [card for card in self.cards if signatures_match(query_signature, card.signature)]
            if query_signature
            else self.cards
        )
        query_tokens = set(tokenize(prompt))
        scored = [
            self._score_card(prompt, query_tokens, query_signature, card)
            for card in candidates
        ]
        scored = [record for record in scored if record["score"] > 0.0]
        return sorted(
            scored,
            key=lambda record: (-float(record["score"]), str(record["card"]["id"])),
        )[:top_k]

    def answer_prompt(self, prompt: str) -> dict[str, Any]:
        matches = self.retrieve(prompt, top_k=3)
        if not matches:
            return {
                "answer": UNKNOWN_ANSWER,
                "retrieved": False,
                "memory_card": None,
                "matches": [],
                "query_signature": prompt_signature(prompt),
            }
        top = matches[0]
        return {
            "answer": top["card"]["target"],
            "retrieved": True,
            "memory_card": top["card"],
            "matches": matches,
            "query_signature": prompt_signature(prompt),
        }

    def evaluate_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        scored: list[dict[str, Any]] = []
        for record in records:
            retrieval = self.answer_prompt(str(record["prompt"]))
            answer = str(retrieval["answer"])
            card = retrieval.get("memory_card")
            scored.append(
                {
                    "id": record["id"],
                    "target": record["target"],
                    "answer": answer,
                    "exact_match": answer == record["target"],
                    "retrieved": bool(retrieval["retrieved"]),
                    "memory_card_id": card["id"] if isinstance(card, dict) else None,
                    "memory_card_source": card["source"] if isinstance(card, dict) else None,
                    "query_signature": retrieval["query_signature"],
                }
            )
        exact = sum(1 for record in scored if record["exact_match"])
        return {
            "count": len(scored),
            "exact": exact,
            "exact_rate": exact / len(scored) if scored else 0.0,
            "retrieved": sum(1 for record in scored if record["retrieved"]),
            "failed_records": [record for record in scored if not record["exact_match"]],
            "records": scored,
        }

    def _score_card(
        self,
        prompt: str,
        query_tokens: set[str],
        query_signature: dict[str, str],
        card: MemoryCard,
    ) -> dict[str, Any]:
        card_tokens = set(tokenize(f"{card.prompt} {card.evidence} {card.target}"))
        overlap = len(query_tokens & card_tokens)
        union = len(query_tokens | card_tokens)
        lexical_score = (overlap / union) if union else 0.0
        exact_prompt_bonus = 2.0 if prompt == card.prompt else 0.0
        signature_bonus = 1.0 if signatures_match(query_signature, card.signature) else 0.0
        return {
            "score": signature_bonus + exact_prompt_bonus + lexical_score,
            "lexical_score": lexical_score,
            "signature_match": signature_bonus > 0.0,
            "exact_prompt_match": exact_prompt_bonus > 0.0,
            "card": asdict(card),
        }
