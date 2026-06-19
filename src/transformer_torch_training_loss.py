"""Shared PyTorch training loss construction for parity probes."""

from __future__ import annotations

from typing import Any

from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_training_state import torch_training_weights_from_state


def build_torch_training_logits(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    context: list[int],
) -> Any:
    """Compute logits from the current trainable tensor state."""

    weights = torch_training_weights_from_state(fixture=fixture, state=state)
    return torch_minimal_logits(
        context,
        {
            "weights": weights,
            "model_config": fixture["model_config"],
        },
        torch,
        runtime,
    )


def build_torch_training_loss_tensor(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    context: list[int],
    target: int,
) -> Any:
    """Build a tensor negative-log-likelihood loss for one microstep."""

    logits = build_torch_training_logits(
        fixture=fixture,
        state=state,
        torch=torch,
        runtime=runtime,
        context=context,
    )
    probabilities = torch.softmax(logits, dim=0)
    return -torch.log(probabilities[target])
