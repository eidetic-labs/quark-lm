"""Answer-level transformer training-step helpers."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar, zero_grad
from neural_char_ops import make_context
from tokenizer import CharTokenizer
from transformer_direct_answer_core import answer_sequence_loss_scalars
from transformer_math import cross_entropy_scalars


def answer_char_loss_scalars(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    position: int,
) -> Scalar:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    context_ids = [*prompt_ids, *target_ids[:position]]
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    return cross_entropy_scalars(model._forward_scalars(context), target_ids[position])

def sampled_choice_candidates(
    target: str,
    candidates: list[str],
    rng: random.Random,
    negative_count: int,
) -> list[str]:
    unique_negatives = sorted({candidate for candidate in candidates if candidate != target})
    if negative_count <= 0:
        selected_negatives: list[str] = []
    elif negative_count >= len(unique_negatives):
        selected_negatives = unique_negatives
    else:
        selected_negatives = rng.sample(unique_negatives, negative_count)
    return [target, *selected_negatives]

def answer_choice_loss_scalars(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    candidates: list[str],
    rng: random.Random,
    negative_count: int,
    max_chars: int = 0,
) -> tuple[Scalar, int]:
    choice_candidates = sampled_choice_candidates(
        example.target,
        candidates,
        rng,
        negative_count,
    )
    scores = [
        -answer_sequence_loss_scalars(
            model,
            tokenizer,
            example.prompt,
            candidate,
            max_chars=max_chars,
        )
        for candidate in choice_candidates
    ]
    return cross_entropy_scalars(scores, target=0), len(choice_candidates)

def train_answer_char(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    learning_rate: float,
) -> float:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    position = rng.randrange(len(target_ids))
    context_ids = [*prompt_ids, *target_ids[:position]]
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    return model.train_step(context, target_ids[position], learning_rate)

def train_answer_char_all_positions(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    learning_rate: float,
) -> float:
    """Train on every target position in one backward (dense next-token signal).

    ``train_answer_char`` samples a single random target position per call, so a
    target of length n receives roughly 1/n of the available gradient signal per
    step. This accumulates the teacher-forced loss across all target positions
    into one backward pass, giving the full per-token signal each step.
    """
    target_ids = tokenizer.encode(example.target)
    if not target_ids:
        return 0.0
    params = model.parameters()
    zero_grad(params)
    loss = Scalar(0.0)
    for position in range(len(target_ids)):
        loss = loss + answer_char_loss_scalars(model, tokenizer, example, position)
    loss = loss / len(target_ids)
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data


def train_answer_mixed_step(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    learning_rate: float,
    candidates: list[str],
    target_loss_weight: float,
    choice_loss_weight: float,
    choice_negatives: int,
    choice_max_chars: int = 0,
) -> dict[str, float]:
    if target_loss_weight <= 0.0 and choice_loss_weight <= 0.0:
        raise ValueError("at least one answer loss weight must be positive")
    params = model.parameters()
    zero_grad(params)
    target_ids = tokenizer.encode(example.target)
    position = rng.randrange(len(target_ids))
    target_loss = answer_char_loss_scalars(model, tokenizer, example, position)
    total_loss = target_loss * target_loss_weight
    choice_loss_value = 0.0
    choice_candidate_count = 0
    if choice_loss_weight > 0.0:
        choice_loss, choice_candidate_count = answer_choice_loss_scalars(
            model,
            tokenizer,
            example,
            candidates,
            rng,
            choice_negatives,
            max_chars=choice_max_chars,
        )
        choice_loss_value = choice_loss.data
        total_loss = total_loss + choice_loss * choice_loss_weight
    total_loss.backward()
    model.apply_gradients(params, learning_rate)
    return {
        "loss": total_loss.data,
        "target_loss": target_loss.data,
        "choice_loss": choice_loss_value,
        "choice_candidate_count": float(choice_candidate_count),
    }


def train_answer_contrast_pair(
    model: Any,
    tokenizer: CharTokenizer,
    in_example: AnswerExample,
    ooc_example: AnswerExample,
    learning_rate: float,
    rng: random.Random,
    max_chars: int = 0,
) -> float:
    """Entity-paired symmetric contrast for closed-world abstention.

    The owner prompt prefers its concrete answer over " unknown."; the
    entity-swapped non-owner prompt prefers " unknown." over that same concrete
    answer. The two prompts differ ONLY in the person, so the preference can only
    flip via the entity tokens -- this trains per-entity conditioning rather than
    shifting the marginal frequency of " unknown." (the failure of plain volume
    augmentation). Reuses the candidate-ranking choice loss, one backward.
    """
    params = model.parameters()
    zero_grad(params)
    in_loss, _ = answer_choice_loss_scalars(
        model, tokenizer, in_example, [ooc_example.target], rng, 1, max_chars
    )
    ooc_loss, _ = answer_choice_loss_scalars(
        model, tokenizer, ooc_example, [in_example.target], rng, 1, max_chars
    )
    total = in_loss + ooc_loss
    total.backward()
    model.apply_gradients(params, learning_rate)
    return total.data
