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

from neural_char_ops import make_context_positioned
from transformer_best_checkpoint import CombinedBestCheckpointTracker
from transformer_no_decay_mask import build_two_group_adamw
from transformer_optimizer import scheduled_learning_rate
from transformer_torch_combined_eval import evaluate_combined_does_both
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


def _example_parts(example: tuple) -> tuple[list[int], list[int] | None, int]:
    """Return (context, abs_positions_or_None, target) for a 2- or 3-tuple example.

    Phase 2 made the canonical example a triple carrying its window's absolute stream
    positions. A legacy 2-tuple (the byte-exact flag-OFF A/B baseline harness) carries
    no positions, so abs_positions is None and the next-token term slot-keys -- correct
    when use_absolute_rope is off, and crashed by the consumption-site guard when on.
    """

    if len(example) == 3:
        context, abs_positions, target = example
        return context, abs_positions, target
    context, target = example
    return context, None, target


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
        # Phase 2: build the ABSOLUTE stream positions for this window and thread them via
        # a per-call dict(runtime) COPY made INSIDE the loop -- never mutate the shared
        # runtime (that would trip the runtime-report parity check and could leak
        # positions across the two entity-paired contrast contexts, which carry
        # independent prompt streams -> independent ids -> independent abs_positions).
        context, abs_positions = make_context_positioned(ids, context_size, pad_id)
        step_runtime = dict(runtime)
        step_runtime["abs_positions"] = abs_positions
        logits = build_torch_training_logits(
            fixture=fixture, state=state, torch=torch, runtime=step_runtime, context=context
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
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError(f"non-finite contrast loss at step {step}")
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
    examples: list[tuple],
    contrast_pairs: list[tuple[Any, Any]],
    steps: int,
    learning_rate: float,
    contrast_weight: float,
    torch: Any,
    runtime: dict[str, Any] | None = None,
    seed: int | None = None,
    shuffle_each_epoch: bool = False,
    validation_probe_paths: list[Any] | None = None,
    eval_every: int = 0,
    eval_responder: Any | None = None,
    f1_floor: float = 0.85,
    gen_floor: float = 0.05,
    eval_max_new_chars: int = 12,
    _eval_report_fn: Any | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """Joint next-token + entity-paired contrast objective on shared weights.

    Each step adds a next-token loss (learn the admitted facts) and a
    contrast-weighted entity-paired loss (owner prefers its concrete answer; the
    entity-swapped non-owner prefers the abstain token), so one model both learns
    facts and acquires entity-conditioned abstention. NaN/Inf-guarded: the contrast
    term is a difference of sequence NLLs, where float underflow could silently flip
    a sign -- fail loud rather than train on garbage.

    When validation_probe_paths and eval_every>0 are supplied, the loop scores the
    live state every eval_every steps and retains the checkpoint with the best gated
    does-both score (CombinedBestCheckpointTracker), restoring it at the end and
    stashing best_combined_score/best_step/best_abstention_f1/best_concrete_gen into
    the returned state. If no step clears both floors it FAILS CLOSED -- prints a
    notice and sets best_combined_score=None, never masquerading as success on the
    last-step weights. All of this is default-OFF: with eval_every=0 the loop and the
    returned state are byte-for-byte identical to the prior behavior.
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

    # Default-OFF combined does-both checkpoint selection. tracker stays None unless a
    # validation slice + eval cadence are supplied, so the parity-validated path is byte
    # identical. _eval_report_fn is a test-only seam injecting a deterministic report.
    tracker = (
        CombinedBestCheckpointTracker(f1_floor, gen_floor)
        if validation_probe_paths and eval_every > 0
        else None
    )

    order = list(range(len(examples)))
    losses: list[float] = []
    grad_norms: list[float] = []
    for step in range(steps):
        position = step % len(examples)
        if shuffle_each_epoch and position == 0:
            order = epoch_shuffle_order(len(examples), seed, step // len(examples))
        # Phase 2 (R-A): this next-token term calls build_torch_training_loss_tensor
        # DIRECTLY -- it does NOT route through _batch_loss (that serves only
        # train_torch_lm, the contrast-OFF path the retrain never hits). So abs_positions
        # must be threaded HERE, via a per-call dict(runtime) COPY, or the fact-learning
        # objective trains slot-keyed while contrast/eval/generation key absolute. A
        # Phase-2 triple (context, abs_positions, target) threads positions; a legacy
        # 2-tuple (the byte-exact flag-OFF A/B baseline harness) carries no positions and
        # slot-keys. Under use_absolute_rope a 2-tuple here would slot-key -> the
        # fail-closed guard at the consumption site crashes it rather than miskey silently.
        context, abs_positions, target = _example_parts(examples[order[position]])
        step_runtime = dict(runtime)
        if abs_positions is not None:
            step_runtime["abs_positions"] = abs_positions
        optimizer.zero_grad()
        total = build_torch_training_loss_tensor(
            fixture=fixture, state=state, torch=torch, runtime=step_runtime,
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
        learning_rate_now = scheduled_learning_rate(
            learning_rate, step + 1,
            warmup_steps=config.get("warmup_steps", 0),
            decay_steps=config.get("decay_steps", 0),
            min_learning_rate=config.get("min_learning_rate", 0.0),
            schedule=config.get("lr_schedule", "linear"),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_now
        optimizer.step()
        losses.append(float(total.detach().cpu()))
        if tracker is not None and (step + 1) % eval_every == 0:
            if _eval_report_fn is not None:
                abstention_f1, concrete_gen = _read_seam_report(_eval_report_fn(step + 1))
            else:
                abstention_f1, concrete_gen = evaluate_combined_does_both(
                    fixture=fixture, state=state, tokenizer=tokenizer, torch=torch,
                    validation_probe_paths=validation_probe_paths,
                    eval_responder=eval_responder, max_new_chars=eval_max_new_chars,
                )
            tracker.consider(
                step + 1, abstention_f1=abstention_f1, concrete_gen=concrete_gen, params=params
            )
    state["grad_norms"] = grad_norms
    if tracker is not None:
        _finalize_combined_best(state, tracker, params)
    return state, losses


def _read_seam_report(report: dict[str, Any]) -> tuple[float, float]:
    """Validate an injected (test-seam) report the same way the real eval bridge does."""

    from transformer_torch_combined_eval import _read_does_both

    return _read_does_both(report)


def _finalize_combined_best(state: dict[str, Any], tracker: Any, params: list[Any]) -> None:
    """Restore the best does-both checkpoint and stash its metrics; fail closed otherwise."""

    if tracker.restore(params):
        state["best_combined_score"] = tracker.best_score
        state["best_step"] = tracker.best_step
        state["best_abstention_f1"] = tracker.best_f1
        state["best_concrete_gen"] = tracker.best_gen
    else:
        # FAIL CLOSED: no checkpoint cleared both floors. Do NOT pass off the last-step
        # weights as a does-both success -- signal absence explicitly.
        state["best_combined_score"] = None
        state["best_step"] = None
        state["best_abstention_f1"] = None
        state["best_concrete_gen"] = None
        print("no does-both checkpoint found")

