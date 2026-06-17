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
    direct_answer_profiled_branch_batch,
)


class TransformerProfiledBranchBatchTest(unittest.TestCase):
    def test_profiled_branch_batch_can_use_baseline_prediction_overrides(
        self,
    ) -> None:
        green = AnswerExample(
            prompt="q: color?\na:",
            target=" green.",
            source="qa:color",
        )
        blue = AnswerExample(
            prompt="q: color?\na:",
            target=" blue.",
            source="qa:color",
        )
        tokenizer = CharTokenizer.train(
            green.prompt + green.target + blue.prompt + blue.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=84,
            )
        )
        model.bout[tokenizer.stoi["g"]].data = 4.0
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            green,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(branch)
        context, target_id, _position = branch
        override_id = tokenizer.stoi["b"]
        current_prediction = max(
            range(tokenizer.vocab_size),
            key=lambda index: model.predict(context)[index],
        )

        batch = direct_answer_profiled_branch_batch(
            model,
            tokenizer,
            green,
            [green, blue],
            random.Random(84),
            branch_position=1,
            batch_size=1,
            terminator=ANSWER_TERMINATOR,
            prediction_overrides={
                (tuple(context), target_id, "qa:color"): override_id,
            },
        )

        self.assertNotEqual(current_prediction, override_id)
        self.assertEqual(batch, [(context, target_id, override_id, "qa:color")])


if __name__ == "__main__":
    unittest.main()
