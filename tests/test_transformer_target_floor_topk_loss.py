from __future__ import annotations

import unittest

from autograd import Scalar
from transformer_lm_branch_retention_floor_loss import retention_floor_loss
from transformer_lm_candidate_set_loss import candidate_set_target_loss
from transformer_lm_target_floor_topk_loss import (
    target_floor_combined_topk_loss,
    target_floor_topk_loss,
)


class TransformerTargetFloorTopKLossTests(unittest.TestCase):
    def test_candidate_set_loss_uses_hard_candidate_count(self) -> None:
        logits = [Scalar(0.0), Scalar(3.0), Scalar(2.0)]

        loss_one = candidate_set_target_loss(
            logits,
            target=0,
            candidate_weight=1.0,
            candidate_count=1,
            vocab_size=3,
        )
        loss_two = candidate_set_target_loss(
            logits,
            target=0,
            candidate_weight=1.0,
            candidate_count=2,
            vocab_size=3,
        )

        self.assertGreater(loss_two.data, loss_one.data)

    def test_target_floor_loss_lifts_anchors_against_hard_candidates(self) -> None:
        model = _FakeModel([Scalar(0.0), Scalar(3.0), Scalar(2.0)])

        loss = target_floor_topk_loss(
            model,
            [([0, 1], 0, 1, "qa")],
            candidate_weight=1.0,
            candidate_count=1,
        )

        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertGreater(loss.data, 0.0)
        self.assertEqual(model.forward_calls, 1)

    def test_target_floor_loss_skips_empty_anchors(self) -> None:
        model = _FakeModel([Scalar(0.0), Scalar(3.0), Scalar(2.0)])

        self.assertIsNone(
            target_floor_topk_loss(
                model,
                [],
                candidate_weight=1.0,
                candidate_count=1,
            )
        )
        self.assertEqual(model.forward_calls, 0)

    def test_combined_loss_matches_floor_plus_candidate_pressure(self) -> None:
        anchors = [([0, 1], 0, 1, "qa")]
        split_model = _FakeModel([Scalar(0.0), Scalar(3.0), Scalar(2.0)])
        combined_model = _FakeModel([Scalar(0.0), Scalar(3.0), Scalar(2.0)])

        split_floor = retention_floor_loss(split_model, anchors, retention_weight=1.0)
        split_topk = target_floor_topk_loss(
            split_model,
            anchors,
            candidate_weight=1.0,
            candidate_count=1,
        )
        combined = target_floor_combined_topk_loss(
            combined_model,
            anchors,
            candidate_weight=1.0,
            candidate_count=1,
            competitor_weight=0.0,
        )

        self.assertIsNotNone(split_floor)
        self.assertIsNotNone(split_topk)
        self.assertIsNotNone(combined)
        assert split_floor is not None
        assert split_topk is not None
        assert combined is not None
        self.assertAlmostEqual(
            combined.data,
            split_floor.data + split_topk.data,
            places=12,
        )
        self.assertEqual(split_model.forward_calls, 2)
        self.assertEqual(combined_model.forward_calls, 1)

    def test_combined_loss_adds_competitor_unlikelihood(self) -> None:
        anchors = [([0, 1], 0, 1, "qa")]
        plain_model = _FakeModel([Scalar(0.0), Scalar(3.0), Scalar(2.0)])
        competitor_model = _FakeModel([Scalar(0.0), Scalar(3.0), Scalar(2.0)])

        plain = target_floor_combined_topk_loss(
            plain_model,
            anchors,
            candidate_weight=1.0,
            candidate_count=1,
            competitor_weight=0.0,
        )
        with_competitor = target_floor_combined_topk_loss(
            competitor_model,
            anchors,
            candidate_weight=1.0,
            candidate_count=1,
            competitor_weight=1.0,
        )

        self.assertIsNotNone(plain)
        self.assertIsNotNone(with_competitor)
        assert plain is not None
        assert with_competitor is not None
        self.assertGreater(with_competitor.data, plain.data)


class _FakeConfig:
    vocab_size = 3


class _FakeModel:
    config = _FakeConfig()

    def __init__(self, logits: list[Scalar]) -> None:
        self._logits = logits
        self.forward_calls = 0

    def _forward_scalars(self, _context: list[int]) -> list[Scalar]:
        self.forward_calls += 1
        return self._logits


if __name__ == "__main__":
    unittest.main()
