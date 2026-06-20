"""Retention-anchor composition for routing-repair objectives."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from replay_plan import BranchReplayRecord
from tokenizer import CharTokenizer
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_eval_profile_retention_anchors import (
    eval_profile_retention_anchor_batch,
)
from transformer_profile_balanced_retention_anchors import (
    profile_balanced_retention_anchor_batch,
)


def routing_repair_retention_anchor_batch(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    eval_records: dict[str, list[dict[str, Any]]] | None,
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[BranchReplayRecord]:
    """Combine training-family and eval-profile retention anchors."""

    training_anchors = profile_balanced_retention_anchor_batch(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    eval_anchors = eval_profile_retention_anchor_batch(
        model,
        tokenizer,
        eval_records,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    return [*training_anchors, *eval_anchors]
