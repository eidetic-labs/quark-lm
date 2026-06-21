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

from pathlib import Path
from typing import Any

from transformer_optimizer import scheduled_learning_rate
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_runtime import (
    TorchGradAccumulator,
    configure_torch_runtime,
    epoch_shuffle_order,
    grad_global_norm,
)
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
    seed: int | None = None,
    shuffle_each_epoch: bool = False,
) -> tuple[dict[str, Any], list[float]]:
    """Train torch tensors over (context, target) examples; return state + losses.

    shuffle_each_epoch reshuffles the example order at each epoch boundary
    (deterministically, keyed by seed+epoch). Default off preserves the fixed
    cyclic order, so the validated single-example parity contract is unchanged.
    """

    if not examples:
        raise ValueError("examples must be non-empty")
    runtime = runtime or {"dtype": "float64", "device": "cpu"}
    configure_torch_runtime(torch, runtime, seed=seed)
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

    accumulator = TorchGradAccumulator(config.get("gradient_accumulation_steps", 1))
    order = list(range(len(examples)))
    losses: list[float] = []
    grad_norms: list[float] = []
    applied_updates = 0
    for step in range(steps):
        position = step % len(examples)
        if shuffle_each_epoch and position == 0:
            order = epoch_shuffle_order(len(examples), seed, step // len(examples))
        context, target = examples[order[position]]
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
        grad_norms.append(grad_global_norm(params, torch))
        if clip and clip > 0.0:
            torch.nn.utils.clip_grad_value_(params, clip)
        # Accumulate clipped grads; apply the mean only at each Nth micro-step,
        # keying the LR schedule on the applied-update count -- mirrors the scalar
        # ScalarOptimizer.apply (sum clipped, mean by pending, LR on update_count).
        # A trailing partial window is left unapplied; N=1 collapses to a step.
        accumulator.add(params)
        if accumulator.ready:
            accumulator.drain_into(params)
            applied_updates += 1
            optimizer.param_groups[0]["lr"] = scheduled_learning_rate(
                learning_rate, applied_updates,
                warmup_steps=config.get("warmup_steps", 0),
                decay_steps=config.get("decay_steps", 0),
                min_learning_rate=config.get("min_learning_rate", 0.0),
            )
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    state["grad_norms"] = grad_norms
    state["applied_updates"] = applied_updates
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


def save_torch_checkpoint(
    path: Path | str,
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write torch-trained weights as a standard checkpoint the eval/spine reads.

    The torch tensors are exported to the scalar weight tree and saved via the
    normal TinyTransformerLM checkpoint format, so a torch-trained model loads
    and scores identically to a scalar-trained one (no separate eval path).
    """

    weights = torch_trained_weights(fixture=fixture, state=state)
    model, _ = TinyTransformerLM.from_dict(
        {"config": fixture["model_config"], "weights": weights}
    )
    destination = Path(path)
    model.save(destination, tokenizer=tokenizer, metadata=metadata)
    return destination


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value
