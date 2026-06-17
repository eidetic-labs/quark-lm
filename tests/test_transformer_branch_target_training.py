from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_branch_diversity_batch,
    direct_answer_lesson,
    train_direct_answer_branch_hidden_projection_margin_unlikelihood,
    train_direct_answer_branch_target_margin_unlikelihood,
    train_direct_answer_branch_target_softmax_unlikelihood,
)


class TransformerBranchTargetTrainingTest(unittest.TestCase):
    def test_branch_target_softmax_improves_restricted_branch_choice(self) -> None:
        fixture = branch_training_fixture(seed=45)
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

        before = restricted_target_probability()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_softmax_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                target_softmax_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(restricted_target_probability(), before)

    def test_branch_target_margin_improves_restricted_logit_gap(self) -> None:
        fixture = branch_training_fixture(seed=42)
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

        def restricted_logit_gap() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                logits = fixture.model._forward_floats(context)
                strongest_other = max(
                    logits[other] for other in branch_targets if other != target
                )
                total += logits[target] - strongest_other
            return total

        before = restricted_logit_gap()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_margin_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.02,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(restricted_logit_gap(), before)

    def test_branch_hidden_projection_margin_improves_projection_gap(self) -> None:
        fixture = branch_training_fixture(seed=43)
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

        def projection(context: list[int], token_id: int) -> float:
            hidden = fixture.model.final_hidden(context)
            output_weights = fixture.model._output_weights_floats()
            return sum(
                hidden[dim] * output_weights[dim][token_id]
                for dim in range(len(hidden))
            )

        def restricted_projection_gap() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                strongest_other = max(
                    projection(context, other)
                    for other in branch_targets
                    if other != target
                )
                total += projection(context, target) - strongest_other
            return total

        before = restricted_projection_gap()
        before_bias = [value.data for value in fixture.model.bout]
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_hidden_projection_margin_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                margin_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(restricted_projection_gap(), before)
        self.assertEqual([value.data for value in fixture.model.bout], before_bias)


if __name__ == "__main__":
    unittest.main()
