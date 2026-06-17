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
