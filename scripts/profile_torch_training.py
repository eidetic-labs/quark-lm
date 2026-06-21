"""Profile a short torch training run to locate the real hot spots.

Phase 0 of the training-optimization plan: confirm the forward is the bottleneck
and quantify the prompt-position-projection cost BEFORE any perf change, so the
vectorization work is evidence-driven rather than guessed. Runs the validated
torch training loop on a representative config (default ctx48 + position
projection -- the slow path) under cProfile and prints the top functions by
cumulative and own time, plus wall-clock per step.

Usage:
    PYTHONPATH=src python scripts/profile_torch_training.py [--steps N] [--dim D]
        [--context-size C] [--no-posproj] [--top K]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neural_char_ops import make_context  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from transformer_model import OptimizationConfig, TransformerConfig  # noqa: E402
from transformer_tiny_lm import TinyTransformerLM  # noqa: E402
from transformer_torch_training_loop import train_torch_lm  # noqa: E402
from transformer_training_parity_fixture import build_scalar_training_parity_fixture  # noqa: E402

RUNTIME = {"dtype": "float64", "device": "cpu"}
SAMPLE = "question: where is mia's ball?\nanswer: under the box.\n" * 4


def build_inputs(*, dim: int, context_size: int, posproj: bool):
    import torch  # noqa: F401  (import cost excluded from the profiled region)

    tokenizer = CharTokenizer.train(SAMPLE)
    ids = tokenizer.encode(SAMPLE)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=context_size,
        embedding_dim=dim,
        feedforward_dim=dim * 2,
        use_prompt_position_projection=posproj,
        seed=17,
    )
    model = TinyTransformerLM.init_random(config)
    pairs = [
        (make_context(ids[:i], context_size, tokenizer.pad_id), ids[i])
        for i in range(1, min(len(ids), 64))
    ]
    fixture = build_scalar_training_parity_fixture(
        fixture_id="profile",
        model=model,
        tokenizer=tokenizer,
        context=pairs[0][0],
        target=pairs[0][1],
        optimizer_config=OptimizationConfig(
            optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0
        ),
        learning_rate=0.01,
        steps=1,
        corpus_hash="profile",
    )
    return fixture, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=48)
    parser.add_argument("--no-posproj", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    import torch

    fixture, pairs = build_inputs(
        dim=args.dim, context_size=args.context_size, posproj=not args.no_posproj
    )
    print(
        f"config: dim={args.dim} ctx={args.context_size} "
        f"posproj={not args.no_posproj} steps={args.steps} pairs={len(pairs)}"
    )

    def run():
        train_torch_lm(
            fixture=fixture,
            examples=pairs,
            steps=args.steps,
            learning_rate=0.01,
            torch=torch,
            runtime=RUNTIME,
        )

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    run()
    profiler.disable()
    elapsed = time.perf_counter() - started
    print(f"wall: {elapsed:.2f}s total, {1000 * elapsed / args.steps:.1f} ms/step\n")

    for sort_key, label in (("cumulative", "BY CUMULATIVE TIME"), ("tottime", "BY OWN TIME")):
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_key)
        stats.print_stats(args.top)
        print(f"=== {label} (top {args.top}) ===")
        # keep the readable rows (path:line(func) + timings), drop pstats preamble
        for line in stream.getvalue().splitlines():
            if "/src/" in line or "{" in line or line.strip().startswith("ncalls"):
                print(line)
        print()


if __name__ == "__main__":
    main()
