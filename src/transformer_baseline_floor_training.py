"""Baseline-floor anchor training steps."""

from __future__ import annotations

import random
from typing import Any

from autograd import Scalar, zero_grad
from replay_plan import BranchReplayRecord, branch_replay_parts
from transformer_baseline_floor_anchor_selection import (
    baseline_floor_objective_anchor_batch,
)
from transformer_direct_modes import BASELINE_FLOOR_REPAIR_STEPS
from transformer_math import cross_entropy_scalars


def train_direct_answer_baseline_floor_anchor_batch(
    model: Any,
    anchors: list[BranchReplayRecord],
    learning_rate: float,
    params: list[Scalar] | None = None,
) -> float:
    if not anchors:
        return 0.0
    params = model.parameters() if params is None else params
    zero_grad(params)
    loss = Scalar(0.0)
    for branch in anchors:
        context, target, _predicted, _profile = branch_replay_parts(branch)
        loss = loss + cross_entropy_scalars(model._forward_scalars(context), target)
    loss = loss / len(anchors)
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data


def train_direct_answer_baseline_floor_anchor_branch_diversity(
    model: Any,
    anchors: list[BranchReplayRecord],
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    params: list[Scalar] | None = None,
) -> float:
    branches: list[tuple[list[int], int, int]] = []
    for branch in anchors:
        context, target, _predicted, _profile = branch_replay_parts(branch)
        probs = model.predict(context)
        predicted = max(range(len(probs)), key=lambda index: probs[index])
        branches.append((context, target, predicted))
    if not branches:
        return 0.0
    return model.train_step_with_branch_diversity(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        contrast_weight,
        params=params,
    )


def train_direct_answer_baseline_floor_anchor_repair(
    model: Any,
    anchors: list[BranchReplayRecord],
    rng: random.Random,
    learning_rate: float,
    batch_size: int,
    params: list[Scalar] | None = None,
) -> float:
    if not anchors:
        return 0.0
    params = model.parameters() if params is None else params
    selected = _target_balanced_repair_examples(anchors, rng, batch_size)
    if not selected:
        return 0.0
    zero_grad(params)
    loss = Scalar(0.0)
    for context, target in selected:
        loss = loss + cross_entropy_scalars(model._forward_scalars(context), target)
    loss = loss / len(selected)
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data


def train_direct_answer_baseline_floor_anchor_repair_stage(
    model: Any,
    anchors: list[BranchReplayRecord],
    rng: random.Random,
    learning_rate: float,
    branch_batch_size: int,
    hard_negative_count: int,
    update_guard: dict[str, Any],
    params: list[Scalar] | None = None,
) -> float:
    repair_loss = 0.0
    repair_batch_size = max(
        branch_batch_size,
        branch_batch_size + max(0, hard_negative_count),
    )
    for _repair_step in range(BASELINE_FLOOR_REPAIR_STEPS):
        update_guard["repair_updates"] += 1
        repair_loss += train_direct_answer_baseline_floor_anchor_repair(
            model,
            anchors,
            rng,
            learning_rate,
            repair_batch_size,
            params=params,
        )
    return repair_loss / max(BASELINE_FLOOR_REPAIR_STEPS, 1)


def train_direct_answer_baseline_floor_stabilization_batch_stage(
    model: Any,
    anchors: list[BranchReplayRecord],
    rng: random.Random,
    learning_rate: float,
    batch_size: int,
    update_guard: dict[str, Any],
    params: list[Scalar] | None = None,
) -> tuple[float, bool]:
    stabilization_batch = baseline_floor_objective_anchor_batch(
        anchors,
        rng,
        batch_size,
    )
    update_guard["stabilization_anchor_batches"] += 1
    update_guard["stabilization_anchor_records"] += len(stabilization_batch)
    return (
        train_direct_answer_baseline_floor_anchor_batch(
            model,
            stabilization_batch,
            learning_rate,
            params=params,
        ),
        bool(stabilization_batch),
    )


def _target_balanced_repair_examples(
    anchors: list[BranchReplayRecord],
    rng: random.Random,
    batch_size: int,
) -> list[tuple[list[int], int]]:
    anchors_by_target: dict[int, list[tuple[list[int], int]]] = {}
    for branch in anchors:
        context, target, _predicted, _profile = branch_replay_parts(branch)
        anchors_by_target.setdefault(target, []).append((context, target))
    target_ids = list(anchors_by_target)
    rng.shuffle(target_ids)
    selected: list[tuple[list[int], int]] = []
    for target_id in target_ids:
        if len(selected) >= max(1, batch_size):
            break
        selected.append(rng.choice(anchors_by_target[target_id]))
    return selected
