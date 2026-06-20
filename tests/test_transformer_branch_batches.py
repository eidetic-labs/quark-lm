from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR, AnswerExample
from support.direct_answer import (
    direct_answer_branch_batch,
    direct_answer_branch_diversity_batch,
    direct_answer_profile_balanced_branch_batch,
    direct_answer_target_balanced_branch_batch,
    direct_answer_target_balanced_branch_diversity_batch,
)


class TransformerBranchBatchTest(unittest.TestCase):
    def test_branch_batch_selects_distinct_branch_targets(self) -> None:
        fixture = branch_training_fixture(seed=40)

        batch = direct_answer_branch_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            random.Random(11),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {fixture.tokenizer.itos[target] for _context, target in batch},
            {"n", "g", "t"},
        )

    def test_target_balanced_branch_batch_samples_rare_targets(self) -> None:
        blue = AnswerExample(prompt="q: blue?\na:", target=" blue.", source="qa:color")
        fixture = branch_training_fixture(seed=40, extra_examples=[blue])
        skewed_examples = [fixture.near for _ in range(20)] + [
            fixture.green,
            fixture.tree,
            blue,
        ]

        batch = direct_answer_target_balanced_branch_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            skewed_examples,
            random.Random(11),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )
        diversity_batch = direct_answer_target_balanced_branch_diversity_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            skewed_examples,
            random.Random(11),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 4)
        self.assertEqual(
            {fixture.tokenizer.itos[target] for _context, target in batch},
            {"n", "g", "t", "b"},
        )
        self.assertEqual(
            {
                fixture.tokenizer.itos[target]
                for _context, target, _predicted in diversity_batch
            },
            {"n", "g", "t", "b"},
        )

    def test_branch_diversity_batch_records_current_predictions(self) -> None:
        fixture = branch_training_fixture(seed=44)
        wrong_id = fixture.tokenizer.stoi["."]
        fixture.model.bout[wrong_id].data = 5.0

        batch = direct_answer_branch_diversity_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            random.Random(14),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {
                fixture.tokenizer.itos[target]
                for _context, target, _predicted in batch
            },
            {"n", "g", "t"},
        )
        self.assertEqual(
            {
                fixture.tokenizer.itos[predicted]
                for _context, _target, predicted in batch
            },
            {"."},
        )

    def test_profile_balanced_branch_batch_covers_training_families(self) -> None:
        self_example = AnswerExample(
            prompt="q: self?\na:",
            target=" near.",
            source="qa:self",
        )
        glossary = AnswerExample(
            prompt="q: glossary?\na:",
            target=" green.",
            source="fact:glossary",
        )
        fixture = branch_training_fixture(
            seed=40,
            extra_examples=[self_example, glossary],
        )

        batch = direct_answer_profile_balanced_branch_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.examples + [self_example, glossary],
            random.Random(11),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 4)
        self.assertEqual(
            {profile for _context, _target, _predicted, profile in batch},
            {"glossary", "owner", "qa", "self"},
        )


if __name__ == "__main__":
    unittest.main()
