"""Transformer generation sampling helpers."""

from __future__ import annotations

import random

from transformer_model import GenerationConfig


def sample_from_probs(probs: list[float], temperature: float, rng: random.Random) -> int:
    adjusted = [pow(max(prob, 1e-12), 1.0 / temperature) for prob in probs]
    total = sum(adjusted)
    threshold = rng.random() * total
    running = 0.0
    for index, prob in enumerate(adjusted):
        running += prob
        if running >= threshold:
            return index
    return len(probs) - 1


def generation_distribution(
    probs: list[float],
    generated_ids: list[int],
    config: GenerationConfig,
) -> list[float]:
    adjusted = list(probs)
    if config.repetition_penalty != 1.0:
        generated = set(generated_ids)
        for token_id in generated:
            adjusted[token_id] = adjusted[token_id] / config.repetition_penalty
    if config.top_k > 0 and config.top_k < len(adjusted):
        keep = set(
            sorted(
                range(len(adjusted)),
                key=lambda index: adjusted[index],
                reverse=True,
            )[: config.top_k]
        )
        adjusted = [
            value if index in keep else 0.0
            for index, value in enumerate(adjusted)
        ]
    if config.top_p < 1.0:
        ranked = sorted(
            range(len(adjusted)),
            key=lambda index: adjusted[index],
            reverse=True,
        )
        keep: set[int] = set()
        cumulative = 0.0
        for index in ranked:
            keep.add(index)
            cumulative += adjusted[index]
            if cumulative >= config.top_p:
                break
        adjusted = [
            value if index in keep else 0.0
            for index, value in enumerate(adjusted)
        ]
    if config.temperature > 0.0 and config.temperature != 1.0:
        adjusted = [
            pow(max(value, 1e-12), 1.0 / config.temperature)
            if value > 0.0
            else 0.0
            for value in adjusted
        ]
    total = sum(adjusted)
    if total <= 0.0:
        return probs
    return [value / total for value in adjusted]
