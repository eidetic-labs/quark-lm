from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_model import (
    AnswerExample,
    answer_training_pool,
    build_model,
    examples_from_sources,
    feature_names,
    glossary_prompt_templates,
    prompt_templates,
)
from curriculum import build_curriculum, read_json


class AnswerModelTest(unittest.TestCase):
    def test_examples_are_derived_from_admitted_facts(self) -> None:
        curriculum = build_curriculum(seed=3)
        grammar = read_json(ROOT / "corpus" / "grammar.json")
        glossary = read_json(ROOT / "corpus" / "glossary.json")
        examples = examples_from_sources(curriculum.train_text, grammar, glossary)
        pairs = {(example.prompt, example.target) for example in examples}

        self.assertIn(("fact place ivy map\nanswer:", " on the shelf."), pairs)
        self.assertIn(("tell me the place of ivy map\nanswer:", " on the shelf."), pairs)
        self.assertIn(("fact color ivy map\nanswer:", " green."), pairs)
        self.assertIn(("which color belongs to ivy map\nanswer:", " green."), pairs)
        self.assertIn(("fact owner map\nanswer:", " ivy."), pairs)
        self.assertIn(("which person has map\nanswer:", " ivy."), pairs)
        self.assertNotIn(("question: where is ivy's map?\nanswer:", " on the shelf."), pairs)
        self.assertNotIn(("question: who has the map?\nanswer:", " ivy."), pairs)
        self.assertNotIn(("ask: color for ivy map\nanswer:", " green."), pairs)
        self.assertIn(("fact place mia ball\nanswer:", " under the box."), pairs)
        self.assertIn(("question: where is mia's ball?\nanswer:", " under the box."), pairs)
        self.assertIn(("question: who has the ball?\nanswer:", " mia."), pairs)
        self.assertIn(("question: where is noah's ball?\nanswer:", " unknown."), pairs)
        self.assertIn(("tell me the place of noah ball\nanswer:", " unknown."), pairs)
        self.assertIn(("which color belongs to mia cup\nanswer:", " unknown."), pairs)
        self.assertIn(("question: who has the water?\nanswer:", " unknown."), pairs)
        self.assertIn(("fact place teacher tree\nanswer:", " near the garden."), pairs)
        self.assertIn(("question: where is teacher's tree?\nanswer:", " near the garden."), pairs)
        self.assertIn(("fact training data teacher tree\nanswer:", " yes."), pairs)
        self.assertIn(
            ("question: is teacher's tree part of your training data?\nanswer:", " yes."),
            pairs,
        )
        self.assertIn(("fact place child bag\nanswer:", " near the shelf."), pairs)
        self.assertIn(("fact owner bag\nanswer:", " child."), pairs)
        self.assertIn(("fact training data child bag\nanswer:", " yes."), pairs)
        self.assertIn(("fact place ivy stone\nanswer:", " under the table."), pairs)
        self.assertIn(("fact color ivy stone\nanswer:", " blue."), pairs)
        self.assertIn(("fact owner stone\nanswer:", " ivy."), pairs)
        self.assertIn(("fact training data ivy stone\nanswer:", " yes."), pairs)
        self.assertIn(("question: where is ivy's stone?\nanswer:", " under the table."), pairs)
        self.assertIn(("question: who has the stone?\nanswer:", " ivy."), pairs)
        self.assertIn(
            ("question: is ivy's stone part of your training data?\nanswer:", " yes."),
            pairs,
        )
        self.assertIn(("question: what are you?\nanswer:", " a closed-world learner."), pairs)
        self.assertIn(("fact self dataset\nanswer:", " the admitted corpus."), pairs)
        self.assertIn(
            ("question: what source guides your self-diagnosis?\nanswer:", " self-improvement reports."),
            pairs,
        )
        self.assertIn(("fact self external_model_shaping\nanswer:", " no."), pairs)
        self.assertIn(
            (
                "question: what happens when you learn something new?\nanswer:",
                " it becomes training data after corpus admission.",
            ),
            pairs,
        )
        self.assertIn(("fact learning weight_update\nanswer:", " weights are updated by training."), pairs)
        self.assertIn(("question: how is the next repair action chosen?\nanswer:", " from report evidence."), pairs)
        self.assertIn(("question: what does corpus mean?\nanswer:", " the admitted training data."), pairs)
        self.assertIn(("define stone\nanswer:", " a small object from the ground."), pairs)
        self.assertIn(("fact glossary language\nanswer:", " words in order."), pairs)

    def test_tiny_model_learns_a_small_answer_set(self) -> None:
        examples = [
            AnswerExample(
                prompt=prompt_templates("mia", "ball", "place")[0],
                target=" under the box.",
                source="test",
            ),
            AnswerExample(
                prompt=prompt_templates("noah", "cup", "place")[0],
                target=" on the table.",
                source="test",
            ),
        ]
        model = build_model(examples, seed=1)
        for _ in range(80):
            for example in examples:
                model.train_step(example, learning_rate=0.1)

        self.assertEqual(model.predict(examples[0].prompt), examples[0].target)
        self.assertEqual(model.predict(examples[1].prompt), examples[1].target)

    def test_semantic_features_bridge_fact_and_question_forms(self) -> None:
        question = set(feature_names("question: where is ivy's map?\nanswer:"))
        fact = set(feature_names("fact place ivy map\nanswer:"))

        self.assertIn("intent:place", question)
        self.assertIn("intent:place", fact)
        self.assertIn("entity:ivy:map", question)
        self.assertIn("entity:ivy:map", fact)
        self.assertIn("intent_entity:place:ivy:map", question)
        self.assertIn("intent_entity:place:ivy:map", fact)

        owner_question = set(feature_names("question: who has the map?\nanswer:"))
        owner_fact = set(feature_names("fact owner map\nanswer:"))
        self.assertIn("intent:owner", owner_question)
        self.assertIn("intent:owner", owner_fact)
        self.assertIn("intent_object:owner:map", owner_question)
        self.assertIn("intent_object:owner:map", owner_fact)

        self_question = set(feature_names("question: what is your dataset?\nanswer:"))
        self_fact = set(feature_names("fact self dataset\nanswer:"))
        self.assertIn("intent:self", self_question)
        self.assertIn("intent:self", self_fact)
        self.assertIn("intent_slot:self:dataset", self_question)
        self.assertIn("intent_slot:self:dataset", self_fact)

        diagnosis_question = set(feature_names("question: what source guides your self-diagnosis?\nanswer:"))
        diagnosis_fact = set(feature_names("fact self diagnosis_source\nanswer:"))
        self.assertIn("intent:self", diagnosis_question)
        self.assertIn("intent:self", diagnosis_fact)
        self.assertIn("intent_slot:self:diagnosis_source", diagnosis_question)
        self.assertIn("intent_slot:self:diagnosis_source", diagnosis_fact)

        admission_question = set(
            feature_names("question: is teacher's tree part of your training data?\nanswer:")
        )
        admission_fact = set(feature_names("fact training data teacher tree\nanswer:"))
        self.assertIn("intent:training_data", admission_question)
        self.assertIn("intent:training_data", admission_fact)
        self.assertIn("intent_entity:training_data:teacher:tree", admission_question)
        self.assertIn("intent_entity:training_data:teacher:tree", admission_fact)

        admission_tag = set(feature_names("training data: teacher tree\nanswer:"))
        self.assertIn("intent:training_data", admission_tag)
        self.assertIn("intent_entity:training_data:teacher:tree", admission_tag)

        place_tell = set(feature_names("tell me the place of teacher tree\nanswer:"))
        self.assertIn("intent:place", place_tell)
        self.assertIn("intent_entity:place:teacher:tree", place_tell)

        color_belongs = set(feature_names("which color belongs to teacher tree\nanswer:"))
        self.assertIn("intent:color", color_belongs)
        self.assertIn("intent_entity:color:teacher:tree", color_belongs)

        glossary_question = set(feature_names("question: what does corpus mean?\nanswer:"))
        glossary_fact = set(feature_names("fact glossary corpus\nanswer:"))
        self.assertIn("intent:glossary", glossary_question)
        self.assertIn("intent:glossary", glossary_fact)
        self.assertIn("intent_word:glossary:corpus", glossary_question)
        self.assertIn("intent_word:glossary:corpus", glossary_fact)

    def test_training_pool_upweights_fact_bridge_examples(self) -> None:
        bridge_place = AnswerExample(
            prompt="tell me the place of child bag\nanswer:",
            target=" near the shelf.",
            source="bridge:place",
        )
        color = AnswerExample(
            prompt="fact color child bag\nanswer:",
            target=" red.",
            source="fact:color",
        )
        unknown = AnswerExample(
            prompt="question: where is noah's ball?\nanswer:",
            target=" unknown.",
            source="unknown:place",
        )

        pool = answer_training_pool([bridge_place, color, unknown])

        self.assertGreater(pool.count(bridge_place), pool.count(unknown))
        self.assertGreater(pool.count(bridge_place), pool.count(color))

        glossary = AnswerExample(
            prompt=glossary_prompt_templates("corpus")[0],
            target=" the admitted training data.",
            source="qa:glossary",
        )
        glossary_pool = answer_training_pool([glossary, unknown])
        self.assertGreater(glossary_pool.count(glossary), glossary_pool.count(unknown))

    def test_summarized_evals_keep_failure_records(self) -> None:
        examples = [
            AnswerExample(
                prompt=prompt_templates("mia", "ball", "place")[0],
                target=" under the box.",
                source="test",
            ),
            AnswerExample(
                prompt=prompt_templates("noah", "cup", "place")[0],
                target=" on the table.",
                source="test",
            ),
        ]
        model = build_model(examples, seed=1)
        for _ in range(80):
            model.train_step(examples[0], learning_rate=0.1)

        from answer_model import summarize_eval

        summary = summarize_eval(
            model,
            [
                {
                    "id": "bad",
                    "prompt": prompt_templates("mia", "ball", "place")[0],
                    "target": " on the table.",
                }
            ],
        )

        self.assertEqual(summary["exact"], 0)
        self.assertEqual(summary["failed_records"][0]["id"], "bad")


if __name__ == "__main__":
    unittest.main()
