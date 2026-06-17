from __future__ import annotations

import random
import unittest

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    direct_answer_branch_context,
    direct_answer_dominant_branch_prediction,
    direct_answer_lesson,
    train_direct_answer_branch_collapse_unlikelihood,
)


class TransformerBranchCollapseRepairsTest(unittest.TestCase):
    def test_dominant_branch_prediction_finds_global_wrong_token(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=38,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        dominant = direct_answer_dominant_branch_prediction(
            model,
            tokenizer,
            [near, green],
            random.Random(8),
            branch_position=1,
            sample_count=0,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(dominant)
        predicted_id, count, scored = dominant
        self.assertEqual(tokenizer.itos[predicted_id], ".")
        self.assertEqual(count, 2)
        self.assertEqual(scored, 2)

    def test_branch_collapse_repair_penalizes_dominant_wrong_token(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=39,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            near,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(branch)
        near_context, near_target, _position = branch
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        before_wrong = model.predict(near_context)[wrong_id]
        before_target = model.predict(near_context)[near_target]
        rng = random.Random(9)

        for _ in range(32):
            train_direct_answer_branch_collapse_unlikelihood(
                model,
                tokenizer,
                near,
                [near, green],
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                sample_count=0,
                terminator=ANSWER_TERMINATOR,
            )

        after_probs = model.predict(near_context)
        self.assertLess(after_probs[wrong_id], before_wrong)
        self.assertGreater(after_probs[near_target], before_target)


if __name__ == "__main__":
    unittest.main()
