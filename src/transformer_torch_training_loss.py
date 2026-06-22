"""Shared PyTorch training loss construction for parity probes."""

from __future__ import annotations

from typing import Any

from transformer_torch_batched_block import batched_logits_fn
from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_profile_support import batched_forward_unsupported_reason
from transformer_torch_training_state import torch_training_weights_from_state


def build_torch_training_logits(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    context: list[int],
) -> Any:
    """Compute logits from the current trainable tensor state.

    Default (use_batched_forward off / unsupported profile): the bit-exact
    per-position ``torch_minimal_logits``. When ``runtime['use_batched_forward']``
    is set AND the profile is batched-supported, route the single context through
    the Tier-2 batched forward (a B=1 batch) instead.
    """

    forward_fixture = {
        "weights": torch_training_weights_from_state(fixture=fixture, state=state),
        "model_config": fixture["model_config"],
    }
    if _use_batched(runtime, fixture["model_config"]):
        return batched_logits_fn(torch, runtime)([context], forward_fixture, torch, runtime)[0]
    return torch_minimal_logits(context, forward_fixture, torch, runtime)


def build_torch_batched_loss_tensor(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    contexts: list[list[int]],
    targets: list[int],
) -> Any:
    """Mean NLL over a batch via one (B, vocab) batched forward.

    Softmax per row (dim=1), gather the per-example target probabilities, mean the
    negative log-likelihoods. Reorders the batch sum vs the per-example stack-sum of
    ``_batch_loss``, so it matches within the validated tolerance band, not bit-exact.
    """

    forward_fixture = {
        "weights": torch_training_weights_from_state(fixture=fixture, state=state),
        "model_config": fixture["model_config"],
    }
    logits = batched_logits_fn(torch, runtime)(contexts, forward_fixture, torch, runtime)
    probabilities = torch.softmax(logits, dim=1)
    target_index = torch.tensor(targets, dtype=torch.long, device=runtime["device"])
    chosen = probabilities.gather(1, target_index.unsqueeze(1)).squeeze(1)
    return (-torch.log(chosen)).mean()


def _use_batched(runtime: dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(runtime.get("use_batched_forward")) and (
        batched_forward_unsupported_reason(config) is None
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
