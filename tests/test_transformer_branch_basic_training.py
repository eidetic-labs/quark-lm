from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR, exclude_scalars
from support.direct_answer import (
    direct_answer_branch_batch,
    direct_answer_branch_diversity_batch,
    direct_answer_lesson,
    train_direct_answer_branch_batch_contrast_unlikelihood,
    train_direct_answer_branch_diversity_unlikelihood,
)


class TransformerBranchBasicTrainingTest(unittest.TestCase):
    def test_branch_batch_contrast_improves_prompt_branch_margin(self) -> None:
        fixture = branch_training_fixture(seed=41)
        batch = direct_answer_branch_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            random.Random(12),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = {target for _context, target in batch}

        def branch_margin() -> float:
            total = 0.0
            for context, target in batch:
                probs = fixture.model.predict(context)
                strongest_other = max(
                    probs[other] for other in branch_targets if other != target
                )
                total += probs[target] - strongest_other
            return total

        before = branch_margin()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(13)

        for _ in range(64):
            train_direct_answer_branch_batch_contrast_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertGreater(branch_margin(), before)

    def test_branch_diversity_unlikelihood_suppresses_global_wrong_token(self) -> None:
        fixture = branch_training_fixture(seed=45)
        wrong_id = fixture.tokenizer.stoi["."]
        fixture.model.bout[wrong_id].data = 5.0
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

        def batch_scores() -> tuple[float, float]:
            wrong_total = 0.0
            target_total = 0.0
            for context, target, _predicted in batch:
                probs = fixture.model.predict(context)
                wrong_total += probs[wrong_id]
                target_total += probs[target]
            return wrong_total, target_total

        before_wrong, before_target = batch_scores()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_diversity_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                contrast_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after_wrong, after_target = batch_scores()
        self.assertLess(after_wrong, before_wrong)
        self.assertGreater(after_target, before_target)

    def test_branch_diversity_training_can_freeze_output_bias(self) -> None:
        fixture = branch_training_fixture(seed=45)
        wrong_id = fixture.tokenizer.stoi["."]
        fixture.model.bout[wrong_id].data = 5.0
        params = exclude_scalars(fixture.model.parameters(), fixture.model.bout)
        before_bout = [value.data for value in fixture.model.bout]
        before_wout = fixture.model.wout[0][wrong_id].data
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )

        train_direct_answer_branch_diversity_unlikelihood(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            lesson,
            random.Random(16),
            learning_rate=0.06,
            negative_weight=1.0,
            positive_weight=1.0,
            contrast_weight=1.0,
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
            params=params,
        )

        self.assertEqual([value.data for value in fixture.model.bout], before_bout)
        self.assertNotEqual(fixture.model.wout[0][wrong_id].data, before_wout)


if __name__ == "__main__":
    unittest.main()
