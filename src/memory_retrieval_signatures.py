"""Prompt signatures for deterministic closed-world memory retrieval."""

from __future__ import annotations

import re


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
