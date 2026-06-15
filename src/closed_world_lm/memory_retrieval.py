"""Deterministic closed-world memory retrieval from the admitted corpus."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .curriculum import DEFAULT_CORPUS_DIR, PROJECT_DIR, read_json, read_jsonl


SCHEMA_VERSION = 1
REPORT_KIND = "retrieval_memory_report"
UNKNOWN_ANSWER = " unknown."
DEFAULT_EVALS = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "paraphrases.jsonl",
    PROJECT_DIR / "evals" / "owner.jsonl",
    PROJECT_DIR / "evals" / "self.jsonl",
    PROJECT_DIR / "evals" / "learning.jsonl",
    PROJECT_DIR / "evals" / "admissions.jsonl",
    PROJECT_DIR / "evals" / "admission_paraphrases.jsonl",
    PROJECT_DIR / "evals" / "glossary.jsonl",
]

TOKEN_RE = re.compile(r"[a-z0-9]+")
WHERE_QUESTION_RE = re.compile(r"question: where is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?")
COLOR_QUESTION_RE = re.compile(
    r"question: what color is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?"
)
OWNER_QUESTION_RE = re.compile(r"question: who has the (?P<object>[a-z]+)\?")
TRAINING_DATA_QUESTION_RE = re.compile(
    r"question: is (?P<person>[a-z]+)'s (?P<object>[a-z]+) part of your training data\?"
)
TRAINING_DATA_TAG_RE = re.compile(r"training data: (?P<person>[a-z]+) (?P<object>[a-z]+)")
PLACE_ASK_RE = re.compile(r"ask: place for (?P<person>[a-z]+) (?P<object>[a-z]+)")
COLOR_ASK_RE = re.compile(r"ask: color for (?P<person>[a-z]+) (?P<object>[a-z]+)")
OWNER_ASK_RE = re.compile(r"ask: owner for (?P<object>[a-z]+)")
PLACE_TELL_RE = re.compile(r"tell me the place of (?P<person>[a-z]+) (?P<object>[a-z]+)")
COLOR_BELONGS_RE = re.compile(r"which color belongs to (?P<person>[a-z]+) (?P<object>[a-z]+)")
OWNER_BELONGS_RE = re.compile(r"which person has (?P<object>[a-z]+)")
GLOSSARY_MEANING_RE = re.compile(r"question: what does (?P<word>[a-z]+) mean\?")
GLOSSARY_DEFINE_RE = re.compile(r"define (?P<word>[a-z]+)")
SELF_PROMPT_PATTERNS = [
    ("kind", re.compile(r"question: what are you\?")),
    ("dataset", re.compile(r"question: what is your dataset\?")),
    ("pretrained_weights", re.compile(r"question: do you use pretrained weights\?")),
    ("unknown_policy", re.compile(r"question: what do you say when a fact is outside your corpus\?")),
    ("improvement_method", re.compile(r"question: how do you improve\?")),
    ("diagnosis_source", re.compile(r"question: what source guides your self-diagnosis\?")),
    ("external_model_shaping", re.compile(r"question: does an external model shape your self-diagnosis\?")),
]
LEARNING_PROMPT_PATTERNS = [
    ("new_data", re.compile(r"question: what happens when you learn something new\?")),
    ("admission", re.compile(r"question: when is something learned\?")),
    ("weight_update", re.compile(r"question: what changes after new training data is admitted\?")),
    ("repair_action", re.compile(r"question: how is the next repair action chosen\?")),
]


@dataclass(frozen=True)
class MemoryCard:
    id: str
    source: str
    profile: str
    prompt: str
    target: str
    evidence: str
    signature: dict[str, str]
    metadata: dict[str, str]


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


def prompt_signature(prompt: str) -> dict[str, str]:
    match = WHERE_QUESTION_RE.search(prompt) or PLACE_ASK_RE.search(prompt) or PLACE_TELL_RE.search(prompt)
    if match:
        return {"intent": "place", "person": match["person"], "object": match["object"]}
    match = COLOR_QUESTION_RE.search(prompt) or COLOR_ASK_RE.search(prompt) or COLOR_BELONGS_RE.search(prompt)
    if match:
        return {"intent": "color", "person": match["person"], "object": match["object"]}
    match = OWNER_QUESTION_RE.search(prompt) or OWNER_ASK_RE.search(prompt) or OWNER_BELONGS_RE.search(prompt)
    if match:
        return {"intent": "owner", "object": match["object"]}
    match = TRAINING_DATA_QUESTION_RE.search(prompt) or TRAINING_DATA_TAG_RE.search(prompt)
    if match:
        return {"intent": "training_data", "person": match["person"], "object": match["object"]}
    match = GLOSSARY_MEANING_RE.search(prompt) or GLOSSARY_DEFINE_RE.search(prompt)
    if match:
        return {"intent": "glossary", "word": match["word"]}
    for slot, pattern in SELF_PROMPT_PATTERNS:
        if pattern.search(prompt):
            return {"intent": "self", "slot": slot}
    for slot, pattern in LEARNING_PROMPT_PATTERNS:
        if pattern.search(prompt):
            return {"intent": "learning", "slot": slot}
    return {}


def signatures_match(query: dict[str, str], candidate: dict[str, str]) -> bool:
    if not query or not candidate:
        return False
    return all(candidate.get(name) == value for name, value in query.items())


def fact_memory_cards(fact: dict[str, Any], source: str) -> list[MemoryCard]:
    person = str(fact["person"])
    obj = str(fact["object"])
    color = str(fact["color"])
    place = f"{fact['relation']} the {fact['container']}"
    fact_id = str(fact["id"])
    evidence = (
        f"{person} has a {color} {obj}; "
        f"{person}'s {obj} is {place}; the {obj} belongs to {person}."
    )
    card_specs = [
        ("place-question", "place", f"question: where is {person}'s {obj}?\nanswer:", f" {place}.", {"intent": "place", "person": person, "object": obj}),
        ("place-tell", "paraphrases", f"tell me the place of {person} {obj}\nanswer:", f" {place}.", {"intent": "place", "person": person, "object": obj}),
        ("place-ask", "admission_paraphrases", f"ask: place for {person} {obj}\nanswer:", f" {place}.", {"intent": "place", "person": person, "object": obj}),
        ("color-question", "color", f"question: what color is {person}'s {obj}?\nanswer:", f" {color}.", {"intent": "color", "person": person, "object": obj}),
        ("color-belongs", "paraphrases", f"which color belongs to {person} {obj}\nanswer:", f" {color}.", {"intent": "color", "person": person, "object": obj}),
        ("color-ask", "admission_paraphrases", f"ask: color for {person} {obj}\nanswer:", f" {color}.", {"intent": "color", "person": person, "object": obj}),
        ("owner-question", "owner", f"question: who has the {obj}?\nanswer:", f" {person}.", {"intent": "owner", "object": obj}),
        ("owner-belongs", "admission_paraphrases", f"which person has {obj}\nanswer:", f" {person}.", {"intent": "owner", "object": obj}),
        ("owner-ask", "admission_paraphrases", f"ask: owner for {obj}\nanswer:", f" {person}.", {"intent": "owner", "object": obj}),
        ("training-data-question", "training_data", f"question: is {person}'s {obj} part of your training data?\nanswer:", " yes.", {"intent": "training_data", "person": person, "object": obj}),
        ("training-data-tag", "admission_paraphrases", f"training data: {person} {obj}\nanswer:", " yes.", {"intent": "training_data", "person": person, "object": obj}),
    ]
    return [
        MemoryCard(
            id=f"{source}:{fact_id}:{suffix}",
            source=source,
            profile=profile,
            prompt=prompt,
            target=target,
            evidence=evidence,
            signature=signature,
            metadata={"fact_id": fact_id, "person": person, "object": obj},
        )
        for suffix, profile, prompt, target, signature in card_specs
    ]


def self_memory_cards(grammar: dict[str, Any]) -> list[MemoryCard]:
    prompts = {
        "kind": "question: what are you?\nanswer:",
        "dataset": "question: what is your dataset?\nanswer:",
        "pretrained_weights": "question: do you use pretrained weights?\nanswer:",
        "unknown_policy": "question: what do you say when a fact is outside your corpus?\nanswer:",
        "improvement_method": "question: how do you improve?\nanswer:",
        "diagnosis_source": "question: what source guides your self-diagnosis?\nanswer:",
        "external_model_shaping": "question: does an external model shape your self-diagnosis?\nanswer:",
    }
    cards: list[MemoryCard] = []
    for fact in grammar.get("self_facts", []):
        slot = str(fact["slot"])
        if slot not in prompts:
            continue
        answer = str(fact["answer"])
        cards.append(
            MemoryCard(
                id=f"corpus:self:{slot}",
                source="corpus:grammar:self_facts",
                profile="self",
                prompt=prompts[slot],
                target=f" {answer}.",
                evidence=f"self {slot} is {answer}.",
                signature={"intent": "self", "slot": slot},
                metadata={"slot": slot},
            )
        )
    return cards


def learning_memory_cards(grammar: dict[str, Any]) -> list[MemoryCard]:
    prompts = {
        "new_data": "question: what happens when you learn something new?\nanswer:",
        "admission": "question: when is something learned?\nanswer:",
        "weight_update": "question: what changes after new training data is admitted?\nanswer:",
        "repair_action": "question: how is the next repair action chosen?\nanswer:",
    }
    cards: list[MemoryCard] = []
    for rule in grammar.get("learning_rules", []):
        slot = str(rule["slot"])
        if slot not in prompts:
            continue
        answer = str(rule["answer"])
        cards.append(
            MemoryCard(
                id=f"corpus:learning:{slot}",
                source="corpus:grammar:learning_rules",
                profile="learning",
                prompt=prompts[slot],
                target=f" {answer}.",
                evidence=f"learning {slot} means {answer}.",
                signature={"intent": "learning", "slot": slot},
                metadata={"slot": slot},
            )
        )
    return cards


def glossary_memory_cards(glossary: dict[str, Any]) -> list[MemoryCard]:
    cards: list[MemoryCard] = []
    for entry in glossary.get("entries", []):
        word = str(entry["word"])
        definition = str(entry["definition"])
        for suffix, prompt in (
            ("meaning", f"question: what does {word} mean?\nanswer:"),
            ("define", f"define {word}\nanswer:"),
        ):
            cards.append(
                MemoryCard(
                    id=f"corpus:glossary:{word}:{suffix}",
                    source="corpus:glossary",
                    profile="glossary",
                    prompt=prompt,
                    target=f" {definition}.",
                    evidence=f"{word}: {definition}.",
                    signature={"intent": "glossary", "word": word},
                    metadata={"word": word},
                )
            )
    return cards


def build_memory_cards(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[MemoryCard]:
    grammar = read_json(corpus_dir / "grammar.json")
    glossary = read_json(corpus_dir / "glossary.json")
    admissions = read_jsonl(corpus_dir / "admissions.jsonl")
    cards: list[MemoryCard] = []
    for fact in grammar.get("story_facts", []):
        cards.extend(fact_memory_cards(fact, "corpus:grammar:story_facts"))
    for fact in admissions:
        cards.extend(fact_memory_cards(fact, "corpus:admissions"))
    cards.extend(self_memory_cards(grammar))
    cards.extend(learning_memory_cards(grammar))
    cards.extend(glossary_memory_cards(glossary))
    return sorted(cards, key=lambda card: card.id)


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


def memory_summary(cards: list[MemoryCard]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    profiles: dict[str, int] = {}
    for card in cards:
        sources[card.source] = sources.get(card.source, 0) + 1
        profiles[card.profile] = profiles.get(card.profile, 0) + 1
    return {
        "card_count": len(cards),
        "source_counts": dict(sorted(sources.items())),
        "profile_counts": dict(sorted(profiles.items())),
    }


def build_retrieval_memory_report(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    eval_paths: list[Path] | None = None,
) -> dict[str, Any]:
    index = ClosedWorldMemoryIndex.from_corpus(corpus_dir)
    evals = {
        path.stem: index.evaluate_records(read_jsonl(path))
        for path in (eval_paths or DEFAULT_EVALS)
        if path.exists()
    }
    total_count = sum(summary["count"] for summary in evals.values())
    total_exact = sum(summary["exact"] for summary in evals.values())
    failed_by_eval = {
        name: [record["id"] for record in summary["failed_records"]]
        for name, summary in evals.items()
        if summary["failed_records"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "corpus_dir": str(corpus_dir),
        "dataset_exclusivity": {
            "uses_external_model": False,
            "external_embeddings": False,
            "pretrained_retriever": False,
            "updates_weights": False,
            "memory_source": "ledgered closed-world corpus",
        },
        "memory": memory_summary(index.cards),
        "evals": evals,
        "summary": {
            "eval_count": len(evals),
            "record_count": total_count,
            "exact": total_exact,
            "exact_rate": total_exact / total_count if total_count else 0.0,
            "failed_by_eval": failed_by_eval,
        },
        "self_improvement": {
            "status": "memory_serves_before_weight_consolidation",
            "rule": (
                "New knowledge can be retrieved immediately after corpus admission; "
                "weight updates remain a gated consolidation step."
            ),
            "next_weight_step": (
                "Use failed retrieval records and neural branch-diversity failures "
                "to decide which memories need consolidation into weights."
            ),
        },
    }


def write_retrieval_memory_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_retrieval_memory_report(args.corpus_dir)
    if args.output is not None:
        write_retrieval_memory_report(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["failed_by_eval"] == {} else 1


if __name__ == "__main__":
    raise SystemExit(main())
