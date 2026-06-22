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

Phase 3 decode mode (KV-cache speedup measurement). Measures per-token decode
wall-clock cache-ON vs cache-OFF, scalar + torch, honestly (bit-exactness is the
deliverable; the speedup is reported, not targeted). The cache turns layer-0
per-step K/V re-projection from O(context_size * dim) into O(dim), so the saving
GROWS WITH context_size and does NOT grow with num_layers (layer-0-only scope --
no depth-compounding). Net-negative for short gens on torch/MPS is an acceptable
honest verdict (launch-overhead bound).

    PYTHONPATH=src .venv/bin/python scripts/profile_torch_training.py --mode decode
        [--decode-engines scalar,torch-cpu,torch-mps] [--context-sweep 16,64,256]
        [--layers-sweep 1,2,4] [--max-new-chars N] [--runs N]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neural_char_ops import make_context, make_context_positioned  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from transformer_model import (  # noqa: E402
    GenerationConfig,
    OptimizationConfig,
    TransformerConfig,
)
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


DECODE_SAMPLE = (
    "mia ball box red cup noah shelf ava leo book pan jug net key "
    "lamp desk door wall roof tree fish bird cake milk\n"
)


def _decode_model(*, dim: int, context_size: int, num_layers: int):
    tokenizer = CharTokenizer.train(DECODE_SAMPLE)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=context_size,
        embedding_dim=dim,
        feedforward_dim=dim * 2,
        attention_heads=2,
        num_layers=num_layers,
        seed=17,
        use_rotary_positions=True,
        use_absolute_rope=True,  # the write-once thesis geometry (cache is valid)
    )
    return tokenizer, TinyTransformerLM.init_random(config)


def _time_scalar_decode(model, tokenizer, *, use_cache, max_new_chars, runs):
    config = GenerationConfig(use_kv_cache=use_cache)
    per_token = []
    for _run in range(runs):
        started = time.perf_counter()
        model.generate_with_trace(tokenizer, "mia", max_new_chars, config)
        per_token.append((time.perf_counter() - started) / max_new_chars)
    return median(per_token) * 1000.0  # ms/token


def _time_torch_decode(model, tokenizer, torch, runtime, *, use_cache, max_new_chars, runs):
    from dataclasses import asdict

    from transformer_torch_minimal_forward import torch_minimal_parity_outputs

    fixture = {
        "weights": model.to_dict()["weights"],
        "model_config": asdict(model.config),
        "forward_cases": [],
        "tokenizer": {
            "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "tokens": list(getattr(tokenizer, "tokens", [])),
        },
    }
    fixture["generation_cases"] = [
        {
            "case_id": "bench",
            "prompt_ids": tokenizer.encode("mia"),
            "max_new_chars": max_new_chars,
            "generation_config": asdict(GenerationConfig(use_kv_cache=use_cache)),
        }
    ]
    per_token = []
    for _run in range(runs):
        started = time.perf_counter()
        torch_minimal_parity_outputs(fixture=fixture, torch=torch, runtime=dict(runtime))
        per_token.append((time.perf_counter() - started) / max_new_chars)
    return median(per_token) * 1000.0  # ms/token


def run_decode_benchmark(args) -> None:
    engines = [e.strip() for e in args.decode_engines.split(",") if e.strip()]
    context_sweep = [int(c) for c in args.context_sweep.split(",")]
    layers_sweep = [int(n) for n in args.layers_sweep.split(",")]

    torch = None
    if any(e.startswith("torch") for e in engines):
        import torch as _torch  # noqa: F401

        torch = _torch

    print(
        f"DECODE benchmark: dim={args.dim} max_new_chars={args.max_new_chars} "
        f"runs={args.runs} warmup={args.warmup}\n"
        "(layer-0-only scope: saving grows with context_size, NOT with num_layers)\n"
    )
    header = f"{'engine':<12}{'ctx':>6}{'layers':>8}{'off ms/tok':>14}{'on ms/tok':>14}{'on/off':>10}"
    print(header)
    print("-" * len(header))

    for engine in engines:
        for num_layers in layers_sweep:
            for context_size in context_sweep:
                tokenizer, model = _decode_model(
                    dim=args.dim, context_size=context_size, num_layers=num_layers
                )
                # Warmup (build/JIT/allocator), excluded from the timed median.
                if engine == "scalar":
                    for _ in range(args.warmup):
                        model.generate_with_trace(
                            tokenizer, "mia", max(1, args.max_new_chars // 4),
                            GenerationConfig(use_kv_cache=True),
                        )
                    off = _time_scalar_decode(
                        model, tokenizer, use_cache=False,
                        max_new_chars=args.max_new_chars, runs=args.runs,
                    )
                    on = _time_scalar_decode(
                        model, tokenizer, use_cache=True,
                        max_new_chars=args.max_new_chars, runs=args.runs,
                    )
                else:
                    device = "mps" if engine == "torch-mps" else "cpu"
                    if device == "mps" and not torch.backends.mps.is_available():
                        print(f"{engine:<12}{context_size:>6}{num_layers:>8}   (MPS unavailable, skipped)")
                        continue
                    dtype = "float32" if device == "mps" else "float64"
                    runtime = {"dtype": dtype, "device": device}
                    for _ in range(args.warmup):
                        _time_torch_decode(
                            model, tokenizer, torch, runtime, use_cache=True,
                            max_new_chars=max(1, args.max_new_chars // 4), runs=1,
                        )
                    off = _time_torch_decode(
                        model, tokenizer, torch, runtime, use_cache=False,
                        max_new_chars=args.max_new_chars, runs=args.runs,
                    )
                    on = _time_torch_decode(
                        model, tokenizer, torch, runtime, use_cache=True,
                        max_new_chars=args.max_new_chars, runs=args.runs,
                    )
                ratio = on / off if off > 0 else float("nan")
                print(
                    f"{engine:<12}{context_size:>6}{num_layers:>8}"
                    f"{off:>14.4f}{on:>14.4f}{ratio:>10.3f}"
                )
    print(
        "\nHonest verdict notes: on/off < 1.0 is a win; > 1.0 is net-negative (the "
        "cache bookkeeping costs more than the saved K/V re-projection, expected for "
        "short gens / small ctx and on launch-overhead-bound MPS). The win should "
        "GROW as ctx grows and stay ~flat across layers (layer-0-only)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("train", "decode"), default="train",
        help="train = cProfile a training run (default); decode = KV-cache speedup",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=48)
    parser.add_argument("--no-posproj", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    # decode-mode flags
    parser.add_argument("--decode-engines", default="scalar,torch-cpu")
    parser.add_argument("--context-sweep", default="16,64,256")
    parser.add_argument("--layers-sweep", default="1,2,4")
    parser.add_argument("--max-new-chars", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "decode":
        run_decode_benchmark(args)
        return

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
