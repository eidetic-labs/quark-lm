from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_branch_diversity_batch,
    direct_answer_lesson,
    train_direct_answer_branch_output_binding_unlikelihood,
    train_direct_answer_branch_rank_margin_unlikelihood,
    train_direct_answer_branch_representation_contrast_unlikelihood,
)


class TransformerBranchTrainingTest(unittest.TestCase):
    def test_branch_representation_contrast_increases_hidden_distance(self) -> None:
        fixture = branch_training_fixture(seed=48)
        batch = direct_answer_branch_diversity_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def average_hidden_distance() -> float:
            distances = []
            for left_index, (left_context, left_target, _left_predicted) in enumerate(
                batch
            ):
                left_hidden = fixture.model.final_hidden(left_context)
                for right_context, right_target, _right_predicted in batch[
                    left_index + 1 :
                ]:
                    if left_target == right_target:
                        continue
                    right_hidden = fixture.model.final_hidden(right_context)
                    distances.append(
                        sum(
                            (left_value - right_value) ** 2
                            for left_value, right_value in zip(
                                left_hidden, right_hidden
                            )
                        )
                    )
            return sum(distances) / len(distances)

        before = average_hidden_distance()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_representation_contrast_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                representation_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(average_hidden_distance(), before)

    def test_branch_output_binding_improves_rank_and_hidden_distance(self) -> None:
        fixture = branch_training_fixture(seed=49)
        fixture.model.bout[fixture.tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})

        def restricted_target_probability() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = fixture.model.predict(context)
                denominator = sum(
                    probs[branch_target] for branch_target in branch_targets
                )
                total += probs[target] / denominator
            return total

        def average_hidden_distance() -> float:
            distances = []
            for left_index, (left_context, left_target, _left_predicted) in enumerate(
                batch
            ):
                left_hidden = fixture.model.final_hidden(left_context)
                for right_context, right_target, _right_predicted in batch[
                    left_index + 1 :
                ]:
                    if left_target == right_target:
                        continue
                    right_hidden = fixture.model.final_hidden(right_context)
                    distances.append(
                        sum(
                            (left_value - right_value) ** 2
                            for left_value, right_value in zip(
                                left_hidden, right_hidden
                            )
                        )
                    )
            return sum(distances) / len(distances)

        before_probability = restricted_target_probability()
        before_distance = average_hidden_distance()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_output_binding_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                binding_weight=2.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(restricted_target_probability(), before_probability)
        self.assertGreater(average_hidden_distance(), before_distance)

    def test_branch_rank_margin_lifts_targets_above_hard_negatives(self) -> None:
        fixture = branch_training_fixture(seed=50)
        fixture.model.bout[fixture.tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def average_target_rank() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = fixture.model.predict(context)
                ranked = sorted(
                    range(len(probs)),
                    key=lambda index: probs[index],
                    reverse=True,
                )
                total += ranked.index(target) + 1
            return total / len(batch)

        before_rank = average_target_rank()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_rank_margin_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertLess(average_target_rank(), before_rank)


if __name__ == "__main__":
    unittest.main()
