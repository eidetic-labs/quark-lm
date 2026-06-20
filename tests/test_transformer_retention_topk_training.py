from __future__ import annotations

import random
import unittest
from typing import Any

from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import direct_answer_branch_diversity_batch


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
            target_floor_anchors=[(context, target, target, "qa")],
        )

        self.assertGreater(loss, 0.0)
        self.assertEqual(calls, [0.03])

    def test_target_floor_training_uses_candidate_set_depth(self) -> None:
        shallow_loss = _target_floor_loss(candidate_count=1)
        full_loss = _target_floor_loss(candidate_count=0)

        self.assertGreater(full_loss, shallow_loss)

    def test_target_floor_training_uses_competitor_pressure(self) -> None:
        plain_loss = _target_floor_loss(candidate_count=1, negative_weight=0.0)
        competitor_loss = _target_floor_loss(candidate_count=1, negative_weight=1.0)

        self.assertGreater(competitor_loss, plain_loss)

    def test_representation_pressure_increases_hidden_distance(self) -> None:
        fixture = branch_training_fixture(seed=76)
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
        before = _average_hidden_distance(fixture.model, batch)

        for _ in range(48):
            fixture.model.train_step_with_branch_retention_topk_softmax(
                batch,
                [],
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                candidate_weight=0.0,
                candidate_count=0,
                representation_weight=1.0,
            )

        self.assertGreater(_average_hidden_distance(fixture.model, batch), before)


def _target_floor_loss(
    candidate_count: int,
    negative_weight: float = 0.0,
) -> float:
    fixture = branch_training_fixture(seed=75)
    model = fixture.model
    context = [fixture.tokenizer.pad_id] * (model.config.context_size - 1)
    context.append(fixture.tokenizer.stoi["q"])
    target = fixture.tokenizer.stoi["n"]

    return model.train_step_with_branch_retention_topk_softmax(
        [],
        [],
        learning_rate=0.0,
        negative_weight=negative_weight,
        positive_weight=0.0,
        candidate_weight=2.0,
        candidate_count=candidate_count,
        target_floor_anchors=[(context, target, target, "qa")],
    )


def _average_hidden_distance(
    model: Any,
    batch: list[tuple[list[int], int, int]],
) -> float:
    distances = []
    for left_index, (left_context, left_target, _left_predicted) in enumerate(batch):
        left_hidden = model.final_hidden(left_context)
        for right_context, right_target, _right_predicted in batch[left_index + 1 :]:
            if left_target == right_target:
                continue
            right_hidden = model.final_hidden(right_context)
            distances.append(
                sum(
                    (left_value - right_value) ** 2
                    for left_value, right_value in zip(left_hidden, right_hidden)
                )
            )
    return sum(distances) / len(distances)


if __name__ == "__main__":
    unittest.main()
