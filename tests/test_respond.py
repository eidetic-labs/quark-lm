from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curriculum import build_curriculum
from respond import CorpusResponder


class RespondTest(unittest.TestCase):
    def test_answers_known_and_heldout_facts_from_corpus(self) -> None:
        curriculum = build_curriculum(seed=3)
        responder = CorpusResponder.train_from_text(curriculum.train_text)

        self.assertEqual(
            responder.answer_prompt("question: where is mia's ball?\nanswer:"),
            " under the box.",
        )
        self.assertEqual(
            responder.answer_prompt("question: what color is ivy's map?\nanswer:"),
            " green.",
        )
        self.assertEqual(
            responder.answer_prompt("ask: place for ivy map\nanswer:"),
            " on the shelf.",
        )
        self.assertEqual(
            responder.answer_prompt("question: who has the map?\nanswer:"),
            " ivy.",
        )
        self.assertEqual(
            responder.answer_prompt("question: where is teacher's tree?\nanswer:"),
            " near the garden.",
        )
        self.assertEqual(
            responder.answer_prompt("question: is teacher's tree part of your training data?\nanswer:"),
            " yes.",
        )
        self.assertEqual(
            responder.answer_prompt("tell me the place of teacher tree\nanswer:"),
            " near the garden.",
        )
        self.assertEqual(
            responder.answer_prompt("which color belongs to teacher tree\nanswer:"),
            " green.",
        )
        self.assertEqual(
            responder.answer_prompt("training data: teacher tree\nanswer:"),
            " yes.",
        )
        self.assertEqual(
            responder.answer_prompt("question: where is child's bag?\nanswer:"),
            " near the shelf.",
        )
        self.assertEqual(
            responder.answer_prompt("question: who has the bag?\nanswer:"),
            " child.",
        )

    def test_unknown_when_fact_is_not_in_corpus(self) -> None:
        curriculum = build_curriculum(seed=3)
        responder = CorpusResponder.train_from_text(curriculum.train_text)

        self.assertEqual(
            responder.answer_prompt("question: where is noah's ball?\nanswer:"),
            " unknown.",
        )
        self.assertEqual(
            responder.answer_prompt("question: who has the water?\nanswer:"),
            " unknown.",
        )
        self.assertEqual(
            responder.answer_prompt("question: is noah's ball part of your training data?\nanswer:"),
            " no.",
        )
        self.assertEqual(
            responder.answer_prompt("define quark\nanswer:"),
            " unknown.",
        )

    def test_answers_operational_self_and_learning_rules(self) -> None:
        curriculum = build_curriculum(seed=3)
        responder = CorpusResponder.train_from_text(curriculum.train_text)

        self.assertEqual(
            responder.answer_prompt("question: what are you?\nanswer:"),
            " a closed-world learner.",
        )
        self.assertEqual(
            responder.answer_prompt("question: what happens when you learn something new?\nanswer:"),
            " it becomes training data after corpus admission.",
        )
        self.assertEqual(
            responder.answer_prompt("question: what source guides your self-diagnosis?\nanswer:"),
            " self-improvement reports.",
        )
        self.assertEqual(
            responder.answer_prompt("question: does an external model shape your self-diagnosis?\nanswer:"),
            " no.",
        )
        self.assertEqual(
            responder.answer_prompt("question: how is the next repair action chosen?\nanswer:"),
            " from report evidence.",
        )

    def test_answers_glossary_definition_questions(self) -> None:
        curriculum = build_curriculum(seed=3)
        responder = CorpusResponder.train_from_text(curriculum.train_text)

        self.assertEqual(
            responder.answer_prompt("question: what does corpus mean?\nanswer:"),
            " the admitted training data.",
        )
        self.assertEqual(
            responder.answer_prompt("define stone\nanswer:"),
            " a small object from the ground.",
        )


if __name__ == "__main__":
    unittest.main()
