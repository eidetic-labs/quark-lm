"""Prompt templates for corpus-derived answer examples."""

from __future__ import annotations


def prompt_templates(
    person: str,
    obj: str,
    kind: str,
    lesson_style: str = "qa",
) -> list[str]:
    if kind == "place":
        if lesson_style == "bridge":
            return [
                f"tell me the place of {person} {obj}\nanswer:",
            ]
        if lesson_style == "fact":
            return [
                f"fact place {person} {obj}\nanswer:",
                f"place fact {person} {obj}\nanswer:",
            ]
        return [
            f"question: where is {person}'s {obj}?\nanswer:",
            f"ask: place for {person} {obj}\nanswer:",
            f"place: {person} {obj}\nanswer:",
        ]
    if kind == "color":
        if lesson_style == "bridge":
            return [
                f"which color belongs to {person} {obj}\nanswer:",
            ]
        if lesson_style == "fact":
            return [
                f"fact color {person} {obj}\nanswer:",
                f"color fact {person} {obj}\nanswer:",
            ]
        return [
            f"question: what color is {person}'s {obj}?\nanswer:",
            f"ask: color for {person} {obj}\nanswer:",
            f"color: {person} {obj}\nanswer:",
        ]
    if kind == "owner":
        if lesson_style == "bridge":
            return [
                f"which person has {obj}\nanswer:",
            ]
        if lesson_style == "fact":
            return [
                f"fact owner {obj}\nanswer:",
                f"owner fact {obj}\nanswer:",
            ]
        return [
            f"question: who has the {obj}?\nanswer:",
            f"ask: owner for {obj}\nanswer:",
            f"owner: {obj}\nanswer:",
        ]
    if kind == "training_data":
        if lesson_style == "fact":
            return [
                f"fact training data {person} {obj}\nanswer:",
                f"training data fact {person} {obj}\nanswer:",
            ]
        return [
            f"question: is {person}'s {obj} part of your training data?\nanswer:",
            f"training data: {person} {obj}\nanswer:",
        ]
    raise ValueError(f"unknown prompt kind {kind!r}")


def self_prompt_templates(slot: str, lesson_style: str = "qa") -> list[str]:
    if lesson_style == "fact":
        return [
            f"fact self {slot}\nanswer:",
            f"self fact {slot}\nanswer:",
        ]
    prompts = {
        "kind": [
            "question: what are you?\nanswer:",
            "ask: self kind\nanswer:",
        ],
        "dataset": [
            "question: what is your dataset?\nanswer:",
            "ask: self dataset\nanswer:",
        ],
        "pretrained_weights": [
            "question: do you use pretrained weights?\nanswer:",
            "ask: self pretrained weights\nanswer:",
        ],
        "unknown_policy": [
            "question: what do you say when a fact is outside your corpus?\nanswer:",
            "ask: self unknown policy\nanswer:",
        ],
        "improvement_method": [
            "question: how do you improve?\nanswer:",
            "ask: self improvement method\nanswer:",
        ],
        "diagnosis_source": [
            "question: what source guides your self-diagnosis?\nanswer:",
            "ask: self diagnosis source\nanswer:",
        ],
        "external_model_shaping": [
            "question: does an external model shape your self-diagnosis?\nanswer:",
            "ask: self external model shaping\nanswer:",
        ],
    }
    return prompts[slot]


def learning_prompt_templates(slot: str, lesson_style: str = "qa") -> list[str]:
    if lesson_style == "fact":
        return [
            f"fact learning {slot}\nanswer:",
            f"learning fact {slot}\nanswer:",
        ]
    prompts = {
        "new_data": [
            "question: what happens when you learn something new?\nanswer:",
            "ask: learning new data\nanswer:",
        ],
        "admission": [
            "question: when is something learned?\nanswer:",
            "ask: learning admission\nanswer:",
        ],
        "weight_update": [
            "question: what changes after new training data is admitted?\nanswer:",
            "ask: learning weight update\nanswer:",
        ],
        "repair_action": [
            "question: how is the next repair action chosen?\nanswer:",
            "ask: learning repair action\nanswer:",
        ],
    }
    return prompts[slot]


def glossary_prompt_templates(word: str, lesson_style: str = "qa") -> list[str]:
    if lesson_style == "fact":
        return [
            f"fact glossary {word}\nanswer:",
            f"glossary fact {word}\nanswer:",
        ]
    return [
        f"question: what does {word} mean?\nanswer:",
        f"define {word}\nanswer:",
    ]
