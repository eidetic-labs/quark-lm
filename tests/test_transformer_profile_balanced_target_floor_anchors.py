from __future__ import annotations

import random
import unittest
from argparse import Namespace

from replay_plan import branch_replay_parts
from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR
from support.direct_answer import (
    direct_answer_lesson,
    direct_answer_profile_balanced_branch_batch,
    train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood,
)
from transformer_profile_balanced_target_floor_anchors import (
    profile_balanced_target_floor_anchor_batch,
    profile_balanced_target_floor_anchors_from_examples,
)
from transformer_routing_repair_batch_evidence import (
    ROUTING_REPAIR_TOPK_BATCH_MODE,
    record_routing_repair_batch_step,
)
from transformer_routing_repair_bundle import (
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
)


class TransformerProfileBalancedTargetFloorAnchorTests(unittest.TestCase):
    def test_target_floor_anchors_preserve_branch_targets(self) -> None:
        fixture = branch_training_fixture(seed=81)
        branches = direct_answer_profile_balanced_branch_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.examples,
            random.Random(5),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )

        anchors = profile_balanced_target_floor_anchor_batch(
            branches,
            random.Random(7),
            batch_size=4,
        )

        self.assertTrue(anchors)
        branch_profiles = _profiles(branches)
        anchor_profiles = _profiles(anchors)
        self.assertTrue(anchor_profiles <= branch_profiles)
        self.assertTrue(
            all(
                target == predicted
                for _context, target, predicted, _profile in map(
                    branch_replay_parts,
                    anchors,
                )
            )
        )

    def test_topk_objective_passes_target_floor_anchors(self) -> None:
        fixture = branch_training_fixture(seed=82)
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        calls: list[list[tuple[list[int], int, int, str]]] = []

        def train_step(
            _branches: list[tuple[list[int], int, int]],
            _retention_anchors: list[tuple[list[int], int, int, str]],
            *_args: object,
            target_floor_anchors: list[tuple[list[int], int, int, str]] | None = None,
            **_kwargs: object,
        ) -> float:
            calls.append(target_floor_anchors or [])
            return 3.25

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
            batch_size=4,
            candidate_count=5,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(loss, 3.25)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0])

    def test_target_floor_from_examples_covers_full_profile_targets(self) -> None:
        fixture = branch_training_fixture(seed=84)
        bounded = direct_answer_profile_balanced_branch_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.examples,
            random.Random(5),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )

        anchors = profile_balanced_target_floor_anchors_from_examples(
            fixture.model,
            fixture.tokenizer,
            fixture.examples,
            random.Random(7),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertGreaterEqual(_target_count(anchors), _target_count(bounded))

    def test_batch_evidence_records_target_floor_counts(self) -> None:
        fixture = branch_training_fixture(seed=83)

        record = record_routing_repair_batch_step(
            args=_topk_args(),
            model=fixture.model,
            tokenizer=fixture.tokenizer,
            branch_examples=fixture.examples,
            rng=random.Random(11),
            direct_step=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertGreater(record["target_floor_anchor_count"], 0)
        self.assertTrue(record["target_floor_anchor_profile_counts"])
        self.assertTrue(record["target_floor_rank_summary"])
        self.assertIn("avg_target_rank", record["target_floor_rank_summary"])


def _profiles(branches: list[tuple[list[int], int, int, str]]) -> set[str]:
    return {branch_replay_parts(branch)[3] for branch in branches}


def _target_count(branches: list[tuple[list[int], int, int, str]]) -> int:
    return len({branch_replay_parts(branch)[1] for branch in branches})


def _topk_args() -> Namespace:
    return Namespace(
        experiment_bundle=PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
        direct_answer_mode=ROUTING_REPAIR_TOPK_BATCH_MODE,
        direct_answer_branch_position=1,
        direct_answer_branch_batch_size=4,
    )


if __name__ == "__main__":
    unittest.main()
