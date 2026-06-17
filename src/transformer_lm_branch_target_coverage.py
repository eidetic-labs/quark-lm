"""Target-set coverage and replay-coverage branch objectives."""

from __future__ import annotations

from autograd import Scalar, zero_grad
from transformer_branch_target_coverage_losses import (
    add_branch_prediction_loss,
    add_target_shares,
    target_balance_loss,
    target_candidate_ids,
    target_set_coverage_loss,
    target_set_mass_from_candidates,
)
from transformer_math import softmax_scalars


class TransformerBranchTargetCoverageMixin:
    def train_step_with_branch_target_set_coverage(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        coverage_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_set = set(branch_targets)
        branch_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            branch_loss = add_branch_prediction_loss(
                branch_loss,
                probs,
                target,
                predicted,
                negative_weight,
                positive_weight,
            )
            if coverage_weight > 0.0 and branch_targets:
                candidate_ids = target_candidate_ids(
                    logits,
                    branch_targets,
                    branch_target_set,
                    self.config.vocab_size,
                    hard_negative_count,
                )
                candidate_probs = softmax_scalars(
                    [logits[candidate_id] for candidate_id in candidate_ids]
                )
                coverage_loss = coverage_loss + target_set_coverage_loss(
                    candidate_probs,
                    candidate_ids,
                    branch_target_set,
                )
        loss = branch_loss / max(len(branches), 1)
        if coverage_weight > 0.0:
            loss = loss + (
                coverage_loss / max(len(branches), 1)
            ) * coverage_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_diversity(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        diversity_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_set = set(branch_targets)
        branch_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        target_share_sums = [Scalar(0.0) for _target in branch_targets]
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            branch_loss = add_branch_prediction_loss(
                branch_loss,
                probs,
                target,
                predicted,
                negative_weight,
                positive_weight,
            )
            if diversity_weight > 0.0 and branch_targets:
                candidate_ids = target_candidate_ids(
                    logits,
                    branch_targets,
                    branch_target_set,
                    self.config.vocab_size,
                    hard_negative_count,
                )
                candidate_probs = softmax_scalars(
                    [logits[candidate_id] for candidate_id in candidate_ids]
                )
                target_set_mass = target_set_mass_from_candidates(
                    candidate_probs,
                    candidate_ids,
                    branch_target_set,
                )
                coverage_loss = coverage_loss + (-(target_set_mass + 1e-12).log())
                add_target_shares(
                    target_share_sums,
                    candidate_probs,
                    target_set_mass,
                )
        loss = branch_loss / max(len(branches), 1)
        if diversity_weight > 0.0 and branch_targets:
            coverage_loss = coverage_loss / max(len(branches), 1)
            diversity_loss = target_balance_loss(
                target_share_sums,
                len(branches),
            )
            loss = loss + ((coverage_loss + diversity_loss) / 2.0) * diversity_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_replay_coverage(
        self,
        branches: list[tuple[list[int], int, int]],
        replay_targets: list[int],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        replay_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        replay_targets = sorted(
            {
                target
                for target in replay_targets
                if 0 <= target < self.config.vocab_size
            }
        )
        if not replay_targets:
            replay_targets = sorted(
                {target for _context, target, _predicted in branches}
            )
        replay_target_set = set(replay_targets)
        branch_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        target_share_sums = [Scalar(0.0) for _target in replay_targets]
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            branch_loss = add_branch_prediction_loss(
                branch_loss,
                probs,
                target,
                predicted,
                negative_weight,
                positive_weight,
            )
            if replay_weight > 0.0 and replay_targets:
                candidate_ids = target_candidate_ids(
                    logits,
                    replay_targets,
                    replay_target_set,
                    self.config.vocab_size,
                    hard_negative_count,
                )
                candidate_probs = softmax_scalars(
                    [logits[candidate_id] for candidate_id in candidate_ids]
                )
                target_set_mass = target_set_mass_from_candidates(
                    candidate_probs,
                    candidate_ids,
                    replay_target_set,
                )
                coverage_loss = coverage_loss + (-(target_set_mass + 1e-12).log())
                add_target_shares(
                    target_share_sums,
                    candidate_probs,
                    target_set_mass,
                )
        loss = branch_loss / max(len(branches), 1)
        if replay_weight > 0.0 and replay_targets:
            coverage_loss = coverage_loss / max(len(branches), 1)
            target_balance = target_balance_loss(
                target_share_sums,
                len(branches),
            )
            loss = loss + ((coverage_loss + target_balance) / 2.0) * replay_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data
