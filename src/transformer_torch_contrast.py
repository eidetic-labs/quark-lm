"""PyTorch entity-paired contrast objective for closed-world abstention.

Torch port of the scalar train_answer_contrast_pair: the owner prompt prefers its
concrete answer over the abstain token; the entity-swapped non-owner prompt
prefers the abstain token over that concrete answer. The two prompts differ only
in the entity, so the preference can only flip via the entity tokens. Runs on the
torch backend so the experiment is feasible at context_size>=48 with
use_prompt_position_projection (where the early entity token reaches the readout)
-- a config the scalar engine could not train in feasible time.

Reuses build_torch_training_logits (differentiable forward) and the AdamW setup;
optimizer state persists across steps (single optimizer for the run).
"""

from __future__ import annotations

from typing import Any

from neural_char_ops import make_context
from transformer_optimizer import scheduled_learning_rate
from transformer_torch_runtime import (
    configure_torch_runtime,
    epoch_shuffle_order,
    grad_global_norm,
)
from transformer_torch_training_loss import (
    build_torch_training_logits,
    build_torch_training_loss_tensor,
)
from transformer_torch_training_state import build_torch_training_state


def torch_answer_sequence_loss(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    prompt_ids: list[int],
    target_ids: list[int],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    """Teacher-forced mean NLL of target_ids given prompt_ids (differentiable)."""

    dtype = getattr(torch, runtime["dtype"])
    if not target_ids:
        return torch.tensor(0.0, dtype=dtype, device=runtime["device"])
    context_size = fixture["model_config"]["context_size"]
    pad_id = fixture["tokenizer"]["pad_id"]
    ids = list(prompt_ids)
    total = torch.tensor(0.0, dtype=dtype, device=runtime["device"])
    for target_id in target_ids:
        context = make_context(ids, context_size, pad_id)
        logits = build_torch_training_logits(
            fixture=fixture, state=state, torch=torch, runtime=runtime, context=context
        )
        probabilities = torch.softmax(logits, dim=0)
        total = total + -torch.log(probabilities[target_id])
        ids.append(target_id)
    return total / len(target_ids)


def torch_answer_choice_loss(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    prompt_ids: list[int],
    candidate_token_lists: list[list[int]],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    """Cross-entropy over candidate sequence scores, preferring candidate 0."""

    scores = [
        -torch_answer_sequence_loss(
            fixture=fixture,
            state=state,
            prompt_ids=prompt_ids,
            target_ids=candidate,
            torch=torch,
            runtime=runtime,
        )
        for candidate in candidate_token_lists
    ]
    probabilities = torch.softmax(torch.stack(scores), dim=0)
    return -torch.log(probabilities[0])


def train_torch_contrast(
    *,
    fixture: dict[str, Any],
    tokenizer: Any,
    pairs: list[tuple[Any, Any]],
    steps: int,
    learning_rate: float,
    torch: Any,
    runtime: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """Train the entity-paired contrast over (in_example, ooc_example) pairs.

    Each step: owner prompt prefers its concrete answer over the abstain token;
    the entity-swapped non-owner prompt prefers the abstain token over the same
    concrete answer. One backward, one optimizer step; AdamW persists across steps.
    """

    if not pairs:
        raise ValueError("pairs must be non-empty")
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

    losses: list[float] = []
    for step in range(steps):
        in_example, ooc_example = pairs[step % len(pairs)]
        in_prompt = tokenizer.encode(in_example.prompt)
        ooc_prompt = tokenizer.encode(ooc_example.prompt)
        concrete = tokenizer.encode(in_example.target)
        abstain = tokenizer.encode(ooc_example.target)

        optimizer.zero_grad()
        in_loss = torch_answer_choice_loss(
            fixture=fixture, state=state, prompt_ids=in_prompt,
            candidate_token_lists=[concrete, abstain], torch=torch, runtime=runtime,
        )
        ooc_loss = torch_answer_choice_loss(
            fixture=fixture, state=state, prompt_ids=ooc_prompt,
            candidate_token_lists=[abstain, concrete], torch=torch, runtime=runtime,
        )
        total = in_loss + ooc_loss
        total.backward()
        if clip and clip > 0.0:
            torch.nn.utils.clip_grad_value_(params, clip)
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    return state, losses


def train_torch_answer_mixed(
    *,
    fixture: dict[str, Any],
    tokenizer: Any,
    examples: list[tuple[list[int], int]],
    contrast_pairs: list[tuple[Any, Any]],
    steps: int,
    learning_rate: float,
    contrast_weight: float,
    torch: Any,
    runtime: dict[str, Any] | None = None,
    seed: int | None = None,
    shuffle_each_epoch: bool = False,
) -> tuple[dict[str, Any], list[float]]:
    """Joint next-token + entity-paired contrast objective on shared weights.

    Each step adds a next-token loss (learn the admitted facts) and a
    contrast-weighted entity-paired loss (owner prefers its concrete answer; the
    entity-swapped non-owner prefers the abstain token), so one model both learns
    facts and acquires entity-conditioned abstention. NaN/Inf-guarded: the contrast
    term is a difference of sequence NLLs, where float underflow could silently flip
    a sign -- fail loud rather than train on garbage.
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

    order = list(range(len(examples)))
    losses: list[float] = []
    grad_norms: list[float] = []
    for step in range(steps):
        position = step % len(examples)
        if shuffle_each_epoch and position == 0:
            order = epoch_shuffle_order(len(examples), seed, step // len(examples))
        context, target = examples[order[position]]
        optimizer.zero_grad()
        total = build_torch_training_loss_tensor(
            fixture=fixture, state=state, torch=torch, runtime=runtime,
            context=context, target=target,
        )
        if contrast_pairs and contrast_weight > 0.0:
            in_example, ooc_example = contrast_pairs[step % len(contrast_pairs)]
            concrete = tokenizer.encode(in_example.target)
            abstain = tokenizer.encode(ooc_example.target)
            in_loss = torch_answer_choice_loss(
                fixture=fixture, state=state, prompt_ids=tokenizer.encode(in_example.prompt),
                candidate_token_lists=[concrete, abstain], torch=torch, runtime=runtime,
            )
            ooc_loss = torch_answer_choice_loss(
                fixture=fixture, state=state, prompt_ids=tokenizer.encode(ooc_example.prompt),
                candidate_token_lists=[abstain, concrete], torch=torch, runtime=runtime,
            )
            total = total + contrast_weight * (in_loss + ooc_loss)
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        total.backward()
        grad_norms.append(grad_global_norm(params, torch))
        if clip and clip > 0.0:
            torch.nn.utils.clip_grad_value_(params, clip)
        optimizer.param_groups[0]["lr"] = scheduled_learning_rate(
            learning_rate, step + 1,
            warmup_steps=config.get("warmup_steps", 0),
            decay_steps=config.get("decay_steps", 0),
            min_learning_rate=config.get("min_learning_rate", 0.0),
        )
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    state["grad_norms"] = grad_norms
    return state, losses

