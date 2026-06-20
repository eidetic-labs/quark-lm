from __future__ import annotations

import unittest

from support.branch_training import branch_training_fixture


class TransformerRetentionTopKTrainingTests(unittest.TestCase):
    def test_retention_topk_training_applies_one_combined_update(self) -> None:
        fixture = branch_training_fixture(seed=74)
        model = fixture.model
        context = [fixture.tokenizer.pad_id] * (model.config.context_size - 1)
        context.append(fixture.tokenizer.stoi["q"])
        target = fixture.tokenizer.stoi["n"]
        predicted = fixture.tokenizer.stoi["g"]
        calls: list[float] = []
        apply_gradients = model.apply_gradients

        def counted_apply_gradients(params: object, learning_rate: float) -> float:
            calls.append(learning_rate)
            return apply_gradients(params, learning_rate)

        model.apply_gradients = counted_apply_gradients

        loss = model.train_step_with_branch_retention_topk_softmax(
            [(context, target, predicted)],
            [(context, target, target, "qa")],
            learning_rate=0.03,
            negative_weight=0.5,
            positive_weight=1.0,
            candidate_weight=2.0,
            candidate_count=3,
        )

        self.assertGreater(loss, 0.0)
        self.assertEqual(calls, [0.03])


if __name__ == "__main__":
    unittest.main()
