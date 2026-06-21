"""Per-type candidate menus + answer-type routing for de-contaminated eval (Phase 3).

Replaces the single global candidate pool (the union of all eval-set targets, with
" unknown." always present, which inflates ranking accuracy and poisons abstention)
with a per-question-type menu drawn from the corpus answer space: a place question
is ranked only against plausible places + the abstain token, not against colors,
owners, or glossary definitions. Answer types and answer strings are derived from
the canonical CorpusResponder regexes/format (`answer_prompt` returns f" {value}."),
so the menus match eval targets by construction. " unknown." is in every menu so
abstention stays measurable per type.
"""

from __future__ import annotations

from corpus_responder import (
    COLOR_ASK_RE,
    COLOR_BELONGS_RE,
    COLOR_QUESTION_RE,
    CorpusResponder,
    GLOSSARY_DEFINE_RE,
    GLOSSARY_MEANING_RE,
    LEARNING_PROMPT_PATTERNS,
    OWNER_ASK_RE,
    OWNER_BELONGS_RE,
    OWNER_QUESTION_RE,
    PLACE_ASK_RE,
    PLACE_TELL_RE,
    SELF_PROMPT_PATTERNS,
    TRAINING_DATA_QUESTION_RE,
    TRAINING_DATA_TAG_RE,
    WHERE_QUESTION_RE,
)

ABSTAIN = " unknown."

_PLACE_PATTERNS = (WHERE_QUESTION_RE, PLACE_ASK_RE, PLACE_TELL_RE)
_COLOR_PATTERNS = (COLOR_QUESTION_RE, COLOR_ASK_RE, COLOR_BELONGS_RE)
_OWNER_PATTERNS = (OWNER_QUESTION_RE, OWNER_ASK_RE, OWNER_BELONGS_RE)
_GLOSSARY_PATTERNS = (GLOSSARY_MEANING_RE, GLOSSARY_DEFINE_RE)


def answer_type_for(prompt: str) -> str | None:
    """Map a probe prompt to its answer type via the canonical responder patterns."""

    if any(pattern.search(prompt) for pattern in _PLACE_PATTERNS):
        return "place"
    if any(pattern.search(prompt) for pattern in _COLOR_PATTERNS):
        return "color"
    if any(pattern.search(prompt) for pattern in _OWNER_PATTERNS):
        return "owner"
    if TRAINING_DATA_QUESTION_RE.search(prompt) or TRAINING_DATA_TAG_RE.search(prompt):
        return "training_data"
    if any(pattern.search(prompt) for pattern in _GLOSSARY_PATTERNS):
        return "glossary"
    if any(pattern.search(prompt) for _slot, pattern in SELF_PROMPT_PATTERNS):
        return "self"
    if any(pattern.search(prompt) for _slot, pattern in LEARNING_PROMPT_PATTERNS):
        return "learning"
    return None


def candidates_by_type(responder: CorpusResponder) -> dict[str, list[str]]:
    """Per-type candidate menus from the corpus answer space (abstain in each)."""

    def menu(values: set[str]) -> list[str]:
        return sorted(values | {ABSTAIN})

    places = {f" {fact.place}." for fact in responder.facts.values() if fact.place}
    colors = {f" {fact.color}." for fact in responder.facts.values() if fact.color}
    owners = {f" {person}." for person, _object in responder.facts}
    glossary = {f" {definition}." for definition in responder.glossary.values()}
    self_answers = {f" {answer}." for answer in responder.self_facts.values()}
    learning = {f" {answer}." for answer in responder.learning_rules.values()}
    return {
        "place": menu(places),
        "color": menu(colors),
        "owner": menu(owners),
        "training_data": [" no.", ABSTAIN, " yes."],
        "glossary": menu(glossary),
        "self": menu(self_answers),
        "learning": menu(learning),
    }


def menu_for(prompt: str, menus: dict[str, list[str]]) -> list[str]:
    """The candidate menu for a probe; abstain-only if the type is unrecognized."""

    answer_type = answer_type_for(prompt)
    if answer_type is None:
        return [ABSTAIN]
    return menus.get(answer_type, [ABSTAIN])
