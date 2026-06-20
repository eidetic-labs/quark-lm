from __future__ import annotations

import random
import unittest
from argparse import Namespace

from replay_plan import branch_replay_parts
from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_lesson,
    train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood,
)
from transformer_eval_profile_retention_anchors import (
    eval_profile_retention_anchor_batch,
)
from transformer_routing_repair_batch_evidence import (
    ROUTING_REPAIR_TOPK_BATCH_MODE,
    record_routing_repair_batch_step,
)
from transformer_routing_repair_bundle import (
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
)


class TransformerEvalProfileRetentionAnchorTests(unittest.TestCase):
    def test_eval_profile_anchor_preserves_represented_target(self) -> None:
        fixture = branch_training_fixture(seed=61)
        _force_prediction(fixture, fixture.near.target[1])

        anchors = eval_profile_retention_anchor_batch(
            fixture.model,
            fixture.tokenizer,
            {"heldout": [fixture.records[0]]},
            random.Random(3),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(_profiles(anchors), {"heldout"})
        self.assertTrue(
            all(
                target == predicted
                for _context, target, predicted, _profile in map(
                    branch_replay_parts,
                    anchors,
                )
            )
        )

    def test_topk_objective_includes_eval_profile_retention_anchors(self) -> None:
        fixture = branch_training_fixture(seed=62)
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        _force_prediction(fixture, fixture.near.target[1])
        calls: list[list[tuple[list[int], int, int, str]]] = []

        def train_step(
            _branches: list[tuple[list[int], int, int]],
            retention_anchors: list[tuple[list[int], int, int, str]],
            *_args: object,
            **_kwargs: object,
        ) -> float:
            calls.append(retention_anchors)
            return 2.5

        fixture.model.train_step_with_branch_retention_topk_softmax = train_step

        loss = train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            lesson,
            random.Random(15),
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=1.0,
            candidate_weight=2.0,
            branch_position=1,
            batch_size=2,
            candidate_count=5,
            terminator=ANSWER_TERMINATOR,
            eval_records={"heldout": [fixture.records[0]]},
        )

        self.assertEqual(loss, 2.5)
        self.assertEqual(len(calls), 1)
        self.assertIn("heldout", _profiles(calls[0]))

    def test_batch_evidence_records_eval_profile_anchor_counts(self) -> None:
        fixture = branch_training_fixture(seed=63)
        _force_prediction(fixture, fixture.near.target[1])

        record = record_routing_repair_batch_step(
            args=_topk_args(),
            model=fixture.model,
            tokenizer=fixture.tokenizer,
            branch_examples=fixture.examples,
            rng=random.Random(11),
            direct_step=1,
            terminator=ANSWER_TERMINATOR,
            eval_records={"heldout": [fixture.records[0]]},
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertIn("heldout", record["retention_anchor_profile_counts"])


def _force_prediction(fixture: object, token: str) -> None:
    token_id = fixture.tokenizer.stoi[token]

    def predict_target(_context: list[int]) -> list[float]:
        probs = [0.0 for _token in fixture.tokenizer.tokens]
        probs[token_id] = 1.0
        return probs

    fixture.model.predict = predict_target


def _profiles(branches: list[tuple[list[int], int, int, str]]) -> set[str]:
    return {branch_replay_parts(branch)[3] for branch in branches}


def _topk_args() -> Namespace:
    return Namespace(
        experiment_bundle=PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
        direct_answer_mode=ROUTING_REPAIR_TOPK_BATCH_MODE,
        direct_answer_branch_position=1,
        direct_answer_branch_batch_size=2,
    )


if __name__ == "__main__":
    unittest.main()
