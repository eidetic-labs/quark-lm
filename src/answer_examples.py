"""Corpus-derived answer examples and prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glossary_probes import glossary_definitions, probe_words
from corpus_responder import CorpusResponder
from answer_prompt_templates import (
    glossary_prompt_templates,
    learning_prompt_templates,
    prompt_templates,
    self_prompt_templates,
)


@dataclass(frozen=True)
class AnswerExample:
    prompt: str
    target: str
    source: str


def examples_from_sources(
    train_text: str,
    grammar: dict[str, Any],
    glossary: dict[str, Any] | None = None,
) -> list[AnswerExample]:
    responder = CorpusResponder.train_from_text(train_text)
    examples: list[AnswerExample] = []
    fact_ids_by_key = {
        (fact["person"], fact["object"]): fact["id"]
        for fact in grammar.get("story_facts", [])
    }
    qa_lesson_ids = set(grammar.get("qa_lesson_ids", []))
    for (person, obj), fact in sorted(responder.facts.items()):
        fact_id = fact_ids_by_key.get((person, obj))
        is_admitted_fact = fact_id is None
        lesson_styles = ["qa", "fact"] if fact_id in qa_lesson_ids or is_admitted_fact else ["fact"]
        answer_lesson_styles = [*lesson_styles, "bridge"]
        if fact.place:
            for lesson_style in answer_lesson_styles:
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {fact.place}.",
                        source=f"{lesson_style}:place",
                    )
                    for prompt in prompt_templates(person, obj, "place", lesson_style)
                )
        if fact.color:
            for lesson_style in answer_lesson_styles:
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {fact.color}.",
                        source=f"{lesson_style}:color",
                    )
                    for prompt in prompt_templates(person, obj, "color", lesson_style)
                )
        if fact.owner:
            for lesson_style in answer_lesson_styles:
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {fact.owner}.",
                        source=f"{lesson_style}:owner",
                    )
                    for prompt in prompt_templates(person, obj, "owner", lesson_style)
                )
        for lesson_style in lesson_styles:
            examples.extend(
                AnswerExample(
                    prompt=prompt,
                    target=" yes.",
                    source=f"{lesson_style}:training_data",
                )
                for prompt in prompt_templates(person, obj, "training_data", lesson_style)
            )

    for fact in grammar.get("unknown_facts", []):
        person = fact["person"]
        obj = fact["object"]
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="unknown:place")
            for prompt in prompt_templates(person, obj, "place")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="bridge:place")
            for prompt in prompt_templates(person, obj, "place", "bridge")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="unknown:color")
            for prompt in prompt_templates(person, obj, "color")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="bridge:color")
            for prompt in prompt_templates(person, obj, "color", "bridge")
        )
        examples.extend(
            AnswerExample(prompt=prompt, target=" no.", source="unknown:training_data")
            for prompt in prompt_templates(person, obj, "training_data")
        )
    for obj in grammar.get("unknown_owner_objects", []):
        examples.extend(
            AnswerExample(prompt=prompt, target=" unknown.", source="unknown:owner")
            for prompt in prompt_templates("", obj, "owner")
        )
    for fact in grammar.get("self_facts", []):
        for lesson_style in ("qa", "fact"):
            examples.extend(
                AnswerExample(
                    prompt=prompt,
                    target=f" {fact['answer']}.",
                    source=f"{lesson_style}:self",
                )
                for prompt in self_prompt_templates(fact["slot"], lesson_style)
            )
    for rule in grammar.get("learning_rules", []):
        for lesson_style in ("qa", "fact"):
            examples.extend(
                AnswerExample(
                    prompt=prompt,
                    target=f" {rule['answer']}.",
                    source=f"{lesson_style}:learning",
                )
                for prompt in learning_prompt_templates(rule["slot"], lesson_style)
            )
    if glossary is not None:
        definitions = glossary_definitions(glossary)
        for word in probe_words(glossary):
            for lesson_style in ("qa", "fact"):
                examples.extend(
                    AnswerExample(
                        prompt=prompt,
                        target=f" {definitions[word]}.",
                        source=f"{lesson_style}:glossary",
                    )
                    for prompt in glossary_prompt_templates(word, lesson_style)
                )
    return examples
