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

from transformer_best_checkpoint import BestCheckpointTracker
from transformer_no_decay_mask import build_two_group_adamw
from transformer_optimizer import scheduled_learning_rate
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_runtime import (
    TorchGradAccumulator,
    configure_torch_runtime,
    epoch_shuffle_order,
    grad_global_norm,
)
from transformer_torch_profile_support import batched_forward_unsupported_reason
from transformer_torch_training_loss import (
    build_torch_batched_loss_tensor,
    build_torch_training_loss_tensor,
)
from transformer_torch_training_state import (
    build_torch_training_state,
    torch_training_weights_from_state,
)


def _batch_loss(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    batch: list[tuple[list[int], int]],
) -> Any:
    """Mean next-token loss over a batch.

    A 1-example batch returns the raw loss with no stack/divide, so batch_size=1
    is bit-exact with the unbatched loop. For B>1 the batch-mean reorders the sum
    (a genuine numerics change validated under the tolerance contract, not 1e-6).
    """

    if len(batch) == 1:
        context, target = batch[0]
        return build_torch_training_loss_tensor(
            fixture=fixture, state=state, torch=torch, runtime=runtime, context=context, target=target
        )
    # Opt-in single-pass batched loss for B>1: only when use_batched_forward is set
    # AND the profile is batched-supported. Flag-off (default) and batch_size==1 keep
    # the byte-for-byte per-example stack-sum path below.
    if runtime.get("use_batched_forward") and (
        batched_forward_unsupported_reason(fixture["model_config"]) is None
    ):
        contexts = [context for context, _ in batch]
        targets = [target for _, target in batch]
        return build_torch_batched_loss_tensor(
            fixture=fixture, state=state, torch=torch, runtime=runtime,
            contexts=contexts, targets=targets,
        )
    per_example = [
        build_torch_training_loss_tensor(
            fixture=fixture, state=state, torch=torch, runtime=runtime, context=context, target=target
        )
        for context, target in batch
    ]
    return torch.stack(per_example).sum() / len(per_example)


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
    validation: list[tuple[list[int], int]] | None = None,
    eval_every: int = 0,
    batch_size: int = 1,
) -> tuple[dict[str, Any], list[float]]:
    """Train torch tensors over (context, target) examples; return state + losses.

    shuffle_each_epoch reshuffles the example order at each epoch boundary
    (deterministically, keyed by seed+epoch). When a validation slice and
    eval_every>0 are supplied, the loop evaluates mean validation loss every
    eval_every micro-steps and restores the best (lowest) checkpoint at the end
    (state['best_validation_loss']/'best_step'). Both default off, so the
    validated single-example parity contract is unchanged.
    """

    if not examples:
        raise ValueError("examples must be non-empty")
    runtime = runtime or {"dtype": "float64", "device": "cpu"}
    configure_torch_runtime(torch, runtime, seed=seed)
    state = build_torch_training_state(fixture=fixture, torch=torch, runtime=runtime)
    params = [parameter["tensor"] for parameter in state["parameters"]]

    config = fixture["optimizer_config"]
    optimizer = build_two_group_adamw(
        state["parameters"],
        learning_rate=learning_rate,
        weight_decay=config["weight_decay"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["epsilon"],
        torch=torch,
        device=runtime["device"],
    )
    clip = config.get("gradient_clip", 0.0)

    accumulator = TorchGradAccumulator(config.get("gradient_accumulation_steps", 1))
    tracker = BestCheckpointTracker() if validation and eval_every > 0 else None
    examples_count = len(examples)
    order = list(range(examples_count))
    current_epoch = -1
    losses: list[float] = []
    grad_norms: list[float] = []
    applied_updates = 0
    for step in range(steps):
        # Disjoint batches of batch_size, advancing through the (optionally
        # reshuffled-per-epoch) order. At batch_size=1 this reduces to the prior
        # `position = step % len` / reshuffle-at-position-0 path exactly.
        epoch = (step * batch_size) // examples_count
        if shuffle_each_epoch and epoch != current_epoch:
            order = epoch_shuffle_order(examples_count, seed, epoch)
            current_epoch = epoch
        batch = [
            examples[order[(step * batch_size + offset) % examples_count]]
            for offset in range(batch_size)
        ]
        optimizer.zero_grad()
        loss = _batch_loss(fixture=fixture, state=state, torch=torch, runtime=runtime, batch=batch)
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
            learning_rate_now = scheduled_learning_rate(
                learning_rate, applied_updates,
                warmup_steps=config.get("warmup_steps", 0),
                decay_steps=config.get("decay_steps", 0),
                min_learning_rate=config.get("min_learning_rate", 0.0),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_now
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if tracker is not None and (step + 1) % eval_every == 0:
            with torch.no_grad():
                validation_loss = sum(
                    eval_torch_loss(
                        fixture=fixture, state=state, context=eval_context,
                        target=eval_target, torch=torch, runtime=runtime,
                    )
                    for eval_context, eval_target in validation
                ) / len(validation)
            tracker.consider(step + 1, validation_loss, params)
    if tracker is not None and tracker.restore(params):
        state["best_validation_loss"] = tracker.best_loss
        state["best_step"] = tracker.best_step
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
