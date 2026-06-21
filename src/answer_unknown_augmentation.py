"""Out-of-corpus -> "unknown" answer-example augmentation.

The corpus declares only a few unknown (person, object) pairs, so the model
memorizes those rather than learning the general "unseen pair -> unknown"
pattern (abstention recall sits at 0). This generates "unknown" answer examples
for every known (person, object) combination that is NOT a real fact and NOT an
already-declared unknown pair.

Leakage: every evaluation probe pair is either a real fact (story_facts: the qa
and heldout sets) or an already-declared unknown pair (unknown_facts: the
unknowns and paraphrase sets). Excluding both means augmented examples never
overlap any eval set. A unit test verifies this against the eval files directly.
"""

from __future__ import annotations

from typing import Any

from answer_examples import AnswerExample
from answer_prompt_templates import prompt_templates


def augment_unknown_examples(grammar: dict[str, Any]) -> list[AnswerExample]:
    """Generate "unknown" examples for safe out-of-corpus (person, object) pairs."""

    story_facts = grammar.get("story_facts", [])
    unknown_facts = grammar.get("unknown_facts", [])
    persons = sorted({fact["person"] for fact in story_facts})
    objects = sorted({fact["object"] for fact in story_facts})
    excluded = {(fact["person"], fact["object"]) for fact in story_facts}
    excluded |= {(fact["person"], fact["object"]) for fact in unknown_facts}

    examples: list[AnswerExample] = []
    for person in persons:
        for obj in objects:
            if (person, obj) in excluded:
                continue
            place_prompts = prompt_templates(person, obj, "place") + prompt_templates(
                person, obj, "place", "bridge"
            )
            color_prompts = prompt_templates(person, obj, "color") + prompt_templates(
                person, obj, "color", "bridge"
            )
            examples.extend(
                AnswerExample(prompt=prompt, target=" unknown.", source="augmented:place")
                for prompt in place_prompts
            )
            examples.extend(
                AnswerExample(prompt=prompt, target=" unknown.", source="augmented:color")
                for prompt in color_prompts
            )
    return examples
