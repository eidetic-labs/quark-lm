"""Prompt feature extraction for closed-world answer selection."""

from __future__ import annotations

import re


WORD_RE = re.compile(r"[a-z']+")
SEMANTIC_FEATURE_WEIGHT = 6
SEMANTIC_PROMPT_PATTERNS = [
    (
        "place",
        re.compile(r"question: where is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?"),
    ),
    (
        "color",
        re.compile(r"question: what color is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?"),
    ),
    ("place", re.compile(r"ask: place for (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"ask: color for (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"place: (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"color: (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"fact place (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"fact color (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"place fact (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"color fact (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("place", re.compile(r"tell me the place of (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("color", re.compile(r"which color belongs to (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("owner", re.compile(r"question: who has the (?P<object>[a-z]+)")),
    ("owner", re.compile(r"ask: owner for (?P<object>[a-z]+)")),
    ("owner", re.compile(r"owner: (?P<object>[a-z]+)")),
    ("owner", re.compile(r"fact owner (?P<object>[a-z]+)")),
    ("owner", re.compile(r"owner fact (?P<object>[a-z]+)")),
    ("owner", re.compile(r"which person has (?P<object>[a-z]+)")),
    (
        "training_data",
        re.compile(r"question: is (?P<person>[a-z]+)'s (?P<object>[a-z]+) part of your training data\?"),
    ),
    ("training_data", re.compile(r"training data: (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("training_data", re.compile(r"fact training data (?P<person>[a-z]+) (?P<object>[a-z]+)")),
    ("training_data", re.compile(r"training data fact (?P<person>[a-z]+) (?P<object>[a-z]+)")),
]
SELF_PROMPT_PATTERNS = [
    ("self", "kind", re.compile(r"question: what are you\?")),
    ("self", "kind", re.compile(r"ask: self kind")),
    ("self", "kind", re.compile(r"fact self kind")),
    ("self", "kind", re.compile(r"self fact kind")),
    ("self", "dataset", re.compile(r"question: what is your dataset\?")),
    ("self", "dataset", re.compile(r"ask: self dataset")),
    ("self", "dataset", re.compile(r"fact self dataset")),
    ("self", "dataset", re.compile(r"self fact dataset")),
    ("self", "pretrained_weights", re.compile(r"question: do you use pretrained weights\?")),
    ("self", "pretrained_weights", re.compile(r"ask: self pretrained weights")),
    ("self", "pretrained_weights", re.compile(r"fact self pretrained_weights")),
    ("self", "pretrained_weights", re.compile(r"self fact pretrained_weights")),
    (
        "self",
        "unknown_policy",
        re.compile(r"question: what do you say when a fact is outside your corpus\?"),
    ),
    ("self", "unknown_policy", re.compile(r"ask: self unknown policy")),
    ("self", "unknown_policy", re.compile(r"fact self unknown_policy")),
    ("self", "unknown_policy", re.compile(r"self fact unknown_policy")),
    ("self", "improvement_method", re.compile(r"question: how do you improve\?")),
    ("self", "improvement_method", re.compile(r"ask: self improvement method")),
    ("self", "improvement_method", re.compile(r"fact self improvement_method")),
    ("self", "improvement_method", re.compile(r"self fact improvement_method")),
    ("self", "diagnosis_source", re.compile(r"question: what source guides your self-diagnosis\?")),
    ("self", "diagnosis_source", re.compile(r"ask: self diagnosis source")),
    ("self", "diagnosis_source", re.compile(r"fact self diagnosis_source")),
    ("self", "diagnosis_source", re.compile(r"self fact diagnosis_source")),
    (
        "self",
        "external_model_shaping",
        re.compile(r"question: does an external model shape your self-diagnosis\?"),
    ),
    ("self", "external_model_shaping", re.compile(r"ask: self external model shaping")),
    ("self", "external_model_shaping", re.compile(r"fact self external_model_shaping")),
    ("self", "external_model_shaping", re.compile(r"self fact external_model_shaping")),
    (
        "learning",
        "new_data",
        re.compile(r"question: what happens when you learn something new\?"),
    ),
    ("learning", "new_data", re.compile(r"ask: learning new data")),
    ("learning", "new_data", re.compile(r"fact learning new_data")),
    ("learning", "new_data", re.compile(r"learning fact new_data")),
    ("learning", "admission", re.compile(r"question: when is something learned\?")),
    ("learning", "admission", re.compile(r"ask: learning admission")),
    ("learning", "admission", re.compile(r"fact learning admission")),
    ("learning", "admission", re.compile(r"learning fact admission")),
    (
        "learning",
        "weight_update",
        re.compile(r"question: what changes after new training data is admitted\?"),
    ),
    ("learning", "weight_update", re.compile(r"ask: learning weight update")),
    ("learning", "weight_update", re.compile(r"fact learning weight_update")),
    ("learning", "weight_update", re.compile(r"learning fact weight_update")),
    (
        "learning",
        "repair_action",
        re.compile(r"question: how is the next repair action chosen\?"),
    ),
    ("learning", "repair_action", re.compile(r"ask: learning repair action")),
    ("learning", "repair_action", re.compile(r"fact learning repair_action")),
    ("learning", "repair_action", re.compile(r"learning fact repair_action")),
]
GLOSSARY_PROMPT_PATTERNS = [
    re.compile(r"question: what does (?P<word>[a-z]+) mean\?"),
    re.compile(r"define (?P<word>[a-z]+)"),
    re.compile(r"fact glossary (?P<word>[a-z]+)"),
    re.compile(r"glossary fact (?P<word>[a-z]+)"),
]


def feature_names(prompt: str) -> list[str]:
    lower = prompt.lower()
    words = WORD_RE.findall(lower)
    names = ["bias"]
    names.extend(f"word:{word}" for word in words)
    names.extend(f"wordpair:{left}:{right}" for left, right in zip(words, words[1:], strict=False))
    names.extend(f"char:{char}" for char in lower)
    for size in (2, 3, 4):
        names.extend(f"ngram{size}:{lower[index:index + size]}" for index in range(len(lower) - size + 1))
    names.extend(semantic_feature_names(lower))
    return names


def semantic_feature_names(lower_prompt: str) -> list[str]:
    names: list[str] = []
    for intent, pattern in SEMANTIC_PROMPT_PATTERNS:
        match = pattern.search(lower_prompt)
        if not match:
            continue
        person = match.groupdict().get("person")
        obj = match["object"]
        semantic = (
            [f"intent:{intent}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"object:{obj}"] * SEMANTIC_FEATURE_WEIGHT
        )
        if person:
            semantic.extend(
                [f"person:{person}"] * SEMANTIC_FEATURE_WEIGHT
                + [f"entity:{person}:{obj}"] * SEMANTIC_FEATURE_WEIGHT
                + [f"intent_entity:{intent}:{person}:{obj}"] * SEMANTIC_FEATURE_WEIGHT
            )
        else:
            semantic.extend([f"intent_object:{intent}:{obj}"] * SEMANTIC_FEATURE_WEIGHT)
        names.extend(semantic)
    for intent, slot, pattern in SELF_PROMPT_PATTERNS:
        if not pattern.search(lower_prompt):
            continue
        names.extend(
            [f"intent:{intent}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"slot:{slot}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"intent_slot:{intent}:{slot}"] * SEMANTIC_FEATURE_WEIGHT
        )
    for pattern in GLOSSARY_PROMPT_PATTERNS:
        match = pattern.search(lower_prompt)
        if not match:
            continue
        word = match["word"]
        names.extend(
            ["intent:glossary"] * SEMANTIC_FEATURE_WEIGHT
            + [f"glossary_word:{word}"] * SEMANTIC_FEATURE_WEIGHT
            + [f"intent_word:glossary:{word}"] * SEMANTIC_FEATURE_WEIGHT
        )
    return names
