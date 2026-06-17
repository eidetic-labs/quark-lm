from __future__ import annotations

import random
import unittest

from support.core import AnswerExample
from support.direct_answer import (
    AnswerCandidateSelector,
    build_answer_selector,
    sampled_choice_candidates,
)


class TransformerAnswerSelectionTest(unittest.TestCase):
    def test_sampled_choice_candidates_keeps_target_first(self) -> None:
        rng = random.Random(4)

        candidates = sampled_choice_candidates(
            " green.",
            [" red.", " green.", " blue.", " red."],
            rng,
            negative_count=1,
        )

        self.assertEqual(candidates[0], " green.")
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[1], " green.")

    def test_answer_candidate_selector_learns_from_closed_world_examples(self) -> None:
        examples = [
            AnswerExample(
                prompt="question: what color is mia's ring?\nanswer:",
                target=" green.",
                source="qa:color",
            ),
            AnswerExample(
                prompt="question: where is mia's ring?\nanswer:",
                target=" in the box.",
                source="qa:place",
            ),
        ]
        selector = build_answer_selector(examples, seed=21)
        candidates = [" green.", " in the box."]

        before = selector.loss(examples[0].prompt, examples[0].target, candidates)
        for _ in range(80):
            for example in examples:
                selector.train_step(example, learning_rate=0.08, candidates=candidates)
        after = selector.loss(examples[0].prompt, examples[0].target, candidates)

        self.assertIsInstance(selector, AnswerCandidateSelector)
        self.assertGreater(before, after)
        self.assertEqual(selector.predict(examples[0].prompt, candidates), " green.")
        self.assertEqual(selector.predict(examples[1].prompt, candidates), " in the box.")


if __name__ == "__main__":
    unittest.main()
