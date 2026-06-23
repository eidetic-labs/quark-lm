"""Shared transformer CLI option groups."""

from __future__ import annotations

import argparse

from transformer_profiles import DEFAULT_PROFILE, MODERN_SMALL_PROFILE


def add_architecture_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transformer-profile",
        choices=[DEFAULT_PROFILE, MODERN_SMALL_PROFILE],
        default=DEFAULT_PROFILE,
        help="Named architecture/optimizer profile for controlled transformer screens.",
    )
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=1)
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument(
        "--use-pre-layer-norm",
        action="store_true",
        help=(
            "Use GPT-style pre-layer normalization in transformer blocks and "
            "apply a final layer norm before the language-model head."
        ),
    )
    parser.add_argument("--use-rms-norm", action="store_true")
    parser.add_argument("--layer-norm-epsilon", type=float, default=1e-5)
    parser.add_argument("--use-gated-mlp", action="store_true")
    parser.add_argument("--tie-output-embeddings", action="store_true")
    parser.add_argument("--use-rotary-positions", action="store_true")
    parser.add_argument("--use-kv-cache-path", action="store_true")
    parser.add_argument(
        "--use-context-mean",
        action="store_true",
        help="Add a mean-pooled context residual to the final transformer representation.",
    )
    parser.add_argument(
        "--use-context-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized projection of the mean-pooled "
            "context to the final transformer representation."
        ),
    )
    parser.add_argument(
        "--use-prompt-prefix-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized projection of non-padding prompt "
            "prefix positions before the final token."
        ),
    )
    parser.add_argument(
        "--use-prompt-position-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized position-specific projection of "
            "non-padding prompt prefix positions before the final token."
        ),
    )
    parser.add_argument(
        "--prompt-position-projection-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the prompt-position projection residual before adding it to "
            "the final representation."
        ),
    )
    parser.add_argument(
        "--use-prompt-attention-summary",
        action="store_true",
        help=(
            "Add a trainable attention-pooled context summary to the final "
            "transformer representation through a zero-initialized projection."
        ),
    )


def add_generation_sampling_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--trace-top-tokens", type=int, default=5)
    parser.add_argument("--use-kv-cache", action="store_true")


def add_optimizer_options(
    parser: argparse.ArgumentParser, *, include_backend: bool = False
) -> None:
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--decay-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    if include_backend:
        parser.add_argument(
            "--backend",
            choices=["scalar_python", "pytorch"],
            default="scalar_python",
            help=(
                "Training backend. 'scalar_python' is the canonical, audited reference "
                "engine; 'pytorch' is the validated performance backend (requires torch)."
            ),
        )
        parser.add_argument(
            "--contrast-weight",
            type=float,
            default=0.0,
            help=(
                "Weight of the entity-paired contrast objective added to the next-token "
                "loss (pytorch backend only). 0 disables it; >0 trains entity-conditioned "
                "abstention jointly with fact learning."
            ),
        )
        parser.add_argument(
            "--batched-forward",
            action="store_true",
            help=(
                "Opt into the Tier-2 tensorized (B,C,D) batched forward on the pytorch "
                "backend (default off; flag-off is byte-exact with the per-position "
                "Tier-1 path). Routes general-LM profiles -- including absolute-RoPE -- "
                "through the fast vectorized path. The slot-keyed prompt-position "
                "projection still fails closed to Tier-1."
            ),
        )


def add_tokenizer_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tokenizer",
        choices=["char", "closed-world-subword"],
        default="char",
        help="Corpus-trained tokenizer to use for this run.",
    )
    parser.add_argument("--tokenizer-manifest", type=str, default=None)
    parser.add_argument("--tokenizer-report", type=str, default=None)
    parser.add_argument("--tokenizer-max-token-chars", type=int, default=4)
    parser.add_argument("--tokenizer-max-new-tokens", type=int, default=32)
