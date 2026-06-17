"""Memory-card construction from the closed-world corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curriculum import DEFAULT_CORPUS_DIR, read_json, read_jsonl


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
