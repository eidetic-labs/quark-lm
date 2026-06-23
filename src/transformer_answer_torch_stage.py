"""Torch-backend core training stage for answer-train (opt-in --backend torch).

The scalar engine remains the canonical, audited reference; this is the validated
PyTorch performance backend (Phase 3a/3b proved exact parity and a checkpoint
bridge). When `answer-train --backend torch` is selected, the core target-loss
stage trains the same from-scratch random init via the torch loop and returns a
TinyTransformerLM carrying the trained weights, so the unchanged completion /
finalization / eval pipeline scores it exactly like a scalar-trained model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from answer_contrast_pairs import build_contrast_pairs
from answer_model import AnswerExample
from corpus_responder import DEFAULT_TRAIN_TEXT, CorpusResponder
from neural_char_ops import make_context_positioned
from tokenizer import CharTokenizer
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_contrast import train_torch_answer_mixed
from transformer_torch_training_loop import torch_trained_weights, train_torch_lm
from transformer_training_parity_fixture import build_scalar_training_parity_fixture

RUNTIME = {"dtype": "float64", "device": "cpu"}

DEFAULT_COMBINED_PROBES = [
    "evals/qa.jsonl",
    "evals/unknowns.jsonl",
    "evals/heldout.jsonl",
    "evals/owner.jsonl",
]


def _combined_best_kwargs(args: argparse.Namespace, tokenizer: CharTokenizer) -> dict[str, Any]:
    """Build the does-both eval machinery (CorpusResponder built ONCE) for the mixed loop."""

    train_text = getattr(args, "train_text", None) or DEFAULT_TRAIN_TEXT
    responder = CorpusResponder.train_from_text(Path(train_text).read_text(encoding="utf-8"))
    probes = getattr(args, "combined_probes", None) or DEFAULT_COMBINED_PROBES
    return {
        "validation_probe_paths": [Path(probe) for probe in probes],
        "eval_every": int(getattr(args, "eval_every_combined", 200)),
        "eval_responder": responder,
        "f1_floor": float(getattr(args, "f1_floor", 0.85)),
        "gen_floor": float(getattr(args, "gen_floor", 0.05)),
    }


def build_torch_answer_training_pairs(
    examples: list[AnswerExample],
    tokenizer: CharTokenizer,
    context_size: int,
    pad_id: int,
) -> list[tuple[list[int], list[int], int]]:
    """Decompose answer examples into all-position teacher-forced triples.

    Phase 2: each pair carries its window's ABSOLUTE stream positions as the middle
    element -- (context, abs_positions, target) -- so the torch training path can key
    RoPE absolutely (Fix A: carry the triple, never reconstruct from the window).
    """

    pairs: list[tuple[list[int], list[int], int]] = []
    for example in examples:
        ids = list(tokenizer.encode(example.prompt))
        for target_id in tokenizer.encode(example.target):
            context, abs_positions = make_context_positioned(ids, context_size, pad_id)
            pairs.append((context, abs_positions, target_id))
            ids.append(target_id)
    return pairs


def train_core_answer_stage_torch(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    training_pool: list[AnswerExample],
    optimizer_config: Any,
) -> Any:
    """Train the core answer stage on the torch backend; return the trained model."""

    import torch  # lazy: torch is an optional performance dependency

    # Opt-in (--batched-forward, default off) Tier-2 batched forward. A per-run dict
    # COPY of the module RUNTIME -- never mutate the shared constant. Default off keeps
    # the per-position Tier-1 path byte-exact; on, the general-LM (projection-off)
    # absolute-RoPE forward runs vectorized. Risk #2: without this no speedup engages.
    runtime = dict(RUNTIME)
    if getattr(args, "batched_forward", False):
        runtime["use_batched_forward"] = True

    pairs = build_torch_answer_training_pairs(
        training_pool, tokenizer, model.config.context_size, tokenizer.pad_id
    )
    if not pairs:
        raise ValueError("answer training pool produced no (context, target) pairs")
    # The torch loop cycles examples sequentially; shuffle (seeded) so a bounded
    # step budget samples the whole corpus instead of only its first lessons.
    random.Random(getattr(args, "seed", 0)).shuffle(pairs)

    first_context, _first_abs, first_target = pairs[0]
    fixture = build_scalar_training_parity_fixture(
        fixture_id="answer-train-torch",
        model=model,
        tokenizer=tokenizer,
        context=first_context,
        target=first_target,
        optimizer_config=optimizer_config,
        learning_rate=args.learning_rate,
        steps=1,  # the parity scalar-train is unused here; torch trains below
        corpus_hash="answer-train-torch",
    )
    contrast_weight = float(getattr(args, "contrast_weight", 0.0) or 0.0)
    if contrast_weight > 0.0:
        grammar = json.loads((args.corpus_dir / "grammar.json").read_text(encoding="utf-8"))
        contrast_pairs = build_contrast_pairs(grammar)
        # Default-OFF does-both checkpoint selection: only when --combined-best is set
        # do we build the CorpusResponder (per-type menus) and pass the eval machinery.
        # Without it the call carries no eval_every/eval_responder, so production
        # answer-train is byte-identical.
        combined_kwargs = _combined_best_kwargs(args, tokenizer) if getattr(args, "combined_best", False) else {}
        state, losses = train_torch_answer_mixed(
            fixture=fixture,
            tokenizer=tokenizer,
            examples=pairs,
            contrast_pairs=contrast_pairs,
            steps=args.steps,
            learning_rate=args.learning_rate,
            contrast_weight=contrast_weight,
            torch=torch,
            runtime=runtime,
            seed=getattr(args, "seed", None),
            **combined_kwargs,
        )
    else:
        state, losses = train_torch_lm(
            fixture=fixture,
            examples=pairs,
            steps=args.steps,
            learning_rate=args.learning_rate,
            torch=torch,
            runtime=runtime,
            seed=getattr(args, "seed", None),
        )
    objective = "next-token+contrast" if contrast_weight > 0.0 else "next-token"
    print(
        f"torch backend ({objective}): trained {args.steps} steps over {len(pairs)} pairs "
        f"train_loss {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    weights = torch_trained_weights(fixture=fixture, state=state)
    trained_model, _ = TinyTransformerLM.from_dict(
        {"config": fixture["model_config"], "weights": weights}
    )
    return trained_model
