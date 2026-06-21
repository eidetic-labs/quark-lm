"""Corpus-driven entity-paired contrast pairs for closed-world abstention training.

Each pair contrasts an admitted fact (owner -> concrete answer) against the same
question with the entity swapped to a NON-owner whose (person, object) pairing is
neither a corpus fact nor a declared unknown (so it is a safe out-of-corpus case,
never an eval probe -> the model must abstain). The two prompts differ only in the
entity, so the answer-vs-unknown preference can flip only via the entity tokens --
the signal the torch contrast objective (train_torch_contrast) optimizes. Withheld
facts are excluded: they are the held-out boundary (measured), not trained.
"""

from __future__ import annotations

from typing import Any

from answer_examples import AnswerExample


def _qa_prompt(person: str, obj: str, kind: str) -> str:
    if kind == "place":
        return f"question: where is {person}'s {obj}?\nanswer:"
    return f"question: what color is {person}'s {obj}?\nanswer:"


def build_contrast_pairs(
    grammar: dict[str, Any],
) -> list[tuple[AnswerExample, AnswerExample]]:
    """Entity-paired (admitted-fact, entity-swapped out-of-corpus) contrast pairs."""

    story_facts = grammar.get("story_facts", [])
    withheld = set(grammar.get("withheld_fact_ids", []))
    persons = sorted({fact["person"] for fact in story_facts})
    excluded = {(fact["person"], fact["object"]) for fact in story_facts}
    excluded |= {
        (fact["person"], fact["object"]) for fact in grammar.get("unknown_facts", [])
    }

    pairs: list[tuple[AnswerExample, AnswerExample]] = []
    for fact in story_facts:
        if fact["id"] in withheld:
            continue
        owner, obj = fact["person"], fact["object"]
        non_owner = next(
            (p for p in persons if p != owner and (p, obj) not in excluded), None
        )
        if non_owner is None:
            continue
        place = f"{fact['relation']} the {fact['container']}"
        for kind, answer in (("place", place), ("color", fact["color"])):
            pairs.append(
                (
                    AnswerExample(_qa_prompt(owner, obj, kind), f" {answer}.", "contrast:in"),
                    AnswerExample(_qa_prompt(non_owner, obj, kind), " unknown.", "contrast:ooc"),
                )
            )
    return pairs
