from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_branch_context,
    direct_answer_lesson,
    train_direct_answer_profile_balanced_branch_rank_collapse_unlikelihood,
)


class TransformerRankCollapseObjectiveTest(unittest.TestCase):
    def test_rank_collapse_objective_penalizes_batch_dominant_wrong_token(self) -> None:
        fixture = branch_training_fixture(seed=59)
        wrong_id = fixture.tokenizer.stoi["."]
        fixture.model.bout[wrong_id].data = 5.0
        branch = direct_answer_branch_context(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(branch)
        context, target_id, _position = branch
        before_wrong = fixture.model.predict(context)[wrong_id]
        before_target = fixture.model.predict(context)[target_id]
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(24):
            train_direct_answer_profile_balanced_branch_rank_collapse_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=1.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=4,
                terminator=ANSWER_TERMINATOR,
            )

        after_probs = fixture.model.predict(context)
        self.assertLess(after_probs[wrong_id], before_wrong)
        self.assertGreater(after_probs[target_id], before_target)


if __name__ == "__main__":
    unittest.main()
