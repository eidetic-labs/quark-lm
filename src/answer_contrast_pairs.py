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

from typing import Any, Callable

from answer_examples import AnswerExample

# scorer(non_owner_prompt, concrete_answer) -> float; higher == HARDER negative
# (the model more wrongly prefers the concrete answer for this non-owner).
HardNegativeScorer = Callable[[str, str], float]


def _qa_prompt(person: str, obj: str, kind: str) -> str:
    if kind == "place":
        return f"question: where is {person}'s {obj}?\nanswer:"
    return f"question: what color is {person}'s {obj}?\nanswer:"


def build_contrast_pairs(
    grammar: dict[str, Any],
    *,
    hard_negative_scorer: HardNegativeScorer | None = None,
) -> list[tuple[AnswerExample, AnswerExample]]:
    """Entity-paired (admitted-fact, entity-swapped out-of-corpus) contrast pairs.

    By default the non-owner is the first eligible person (deterministic). With a
    hard_negative_scorer, the HARDEST eligible non-owner is chosen instead -- the
    one the model most wrongly prefers the concrete answer for -- which sharpens
    the contrast margin. Selection only ranks the already-eligible (leakage-safe)
    non-owners, so the partition guarantees are unchanged; scorer=None is
    byte-for-byte identical to the prior behavior.
    """

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
        eligible = [p for p in persons if p != owner and (p, obj) not in excluded]
        if not eligible:
            continue
        place = f"{fact['relation']} the {fact['container']}"
        if hard_negative_scorer is None:
            non_owner = eligible[0]
        else:
            non_owner = max(
                eligible,
                key=lambda person: hard_negative_scorer(_qa_prompt(person, obj, "place"), f" {place}."),
            )
        for kind, answer in (("place", place), ("color", fact["color"])):
            pairs.append(
                (
                    AnswerExample(_qa_prompt(owner, obj, kind), f" {answer}.", "contrast:in"),
                    AnswerExample(_qa_prompt(non_owner, obj, kind), " unknown.", "contrast:ooc"),
                )
            )
    return pairs
