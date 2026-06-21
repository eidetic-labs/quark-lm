"""Minimal PyTorch corpus training loop (Phase 3a).

Composes the validated torch forward/backward/optimizer components into a loop
that trains over a list of (context, target) examples. The torch tensors are
initialized from a from-scratch scalar fixture's initial random weights -- no
pretrained weights. With gradient_accumulation_steps=1 and no LR schedule, this
loop reproduces the scalar reference's training step-for-step, so torch can be
validated as a faithful (and, at scale, faster) backend.

This is deliberately a thin orchestrator over already-parity-tested pieces:
build_torch_training_state, build_torch_training_loss_tensor, torch.optim.AdamW.
"""

from __future__ import annotations

from typing import Any

from transformer_torch_tensor_ops import torch_to_list
from transformer_torch_training_loss import build_torch_training_loss_tensor
from transformer_torch_training_state import (
    build_torch_training_state,
    torch_training_weights_from_state,
)


def train_torch_lm(
    *,
    fixture: dict[str, Any],
    examples: list[tuple[list[int], int]],
    steps: int,
    learning_rate: float,
    torch: Any,
    runtime: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """Train torch tensors over (context, target) examples; return state + losses."""

    if not examples:
        raise ValueError("examples must be non-empty")
    runtime = runtime or {"dtype": "float64", "device": "cpu"}
    state = build_torch_training_state(fixture=fixture, torch=torch, runtime=runtime)
    params = [parameter["tensor"] for parameter in state["parameters"]]

    config = fixture["optimizer_config"]
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        betas=(config["beta1"], config["beta2"]),
        eps=config["epsilon"],
        weight_decay=config["weight_decay"],
    )
    clip = config.get("gradient_clip", 0.0)

    losses: list[float] = []
    for step in range(steps):
        context, target = examples[step % len(examples)]
        optimizer.zero_grad()
        loss = build_torch_training_loss_tensor(
            fixture=fixture,
            state=state,
            torch=torch,
            runtime=runtime,
            context=context,
            target=target,
        )
        loss.backward()
        if clip and clip > 0.0:
            torch.nn.utils.clip_grad_value_(params, clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return state, losses


def eval_torch_loss(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    context: list[int],
    target: int,
    torch: Any,
    runtime: dict[str, Any] | None = None,
) -> float:
    """Teacher-forced NLL of (context -> target) for the current torch state."""

    runtime = runtime or {"dtype": "float64", "device": "cpu"}
    loss = build_torch_training_loss_tensor(
        fixture=fixture, state=state, torch=torch, runtime=runtime, context=context, target=target
    )
    return float(loss.detach().cpu())


def torch_trained_weights(*, fixture: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Export the trained torch tensors back to the scalar checkpoint weight tree."""

    weights = torch_training_weights_from_state(fixture=fixture, state=state)
    return _serialize(weights)


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if hasattr(value, "detach"):
        return torch_to_list(value)
    return value
