"""Transformer model configuration and checkpoint metadata surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TRANSFORMER_ARCHITECTURE = "tiny-decoder-only-transformer"
TRANSFORMER_CHECKPOINT_FORMAT = "quarklm-transformer-v2"
TRANSFORMER_TOKENIZER = "tokenizer.CharTokenizer"


@dataclass
class TransformerConfig:
    vocab_size: int
    context_size: int = 16
    embedding_dim: int = 8
    feedforward_dim: int = 16
    seed: int = 17
    num_layers: int = 1
    attention_heads: int = 1
    use_layer_norm: bool = False
    use_pre_layer_norm: bool = False
    use_rms_norm: bool = False
    layer_norm_epsilon: float = 1e-5
    use_gated_mlp: bool = False
    tie_output_embeddings: bool = False
    use_rotary_positions: bool = False
    use_kv_cache_path: bool = False
    use_context_mean: bool = False
    use_context_projection: bool = False
    use_prompt_prefix_projection: bool = False
    use_prompt_position_projection: bool = False
    prompt_position_projection_scale: float = 1.0
    use_prompt_attention_summary: bool = False


@dataclass
class OptimizationConfig:
    optimizer: str = "sgd"
    gradient_clip: float = 5.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    warmup_steps: int = 0
    decay_steps: int = 0
    min_learning_rate: float = 0.0
    gradient_accumulation_steps: int = 1


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    trace_top_tokens: int = 5
    use_kv_cache: bool = False


def validate_transformer_config(config: TransformerConfig) -> None:
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.context_size <= 0:
        raise ValueError("context_size must be positive")
    if config.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if config.feedforward_dim <= 0:
        raise ValueError("feedforward_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.attention_heads <= 0:
        raise ValueError("attention_heads must be positive")
    if config.embedding_dim % config.attention_heads != 0:
        raise ValueError("embedding_dim must be divisible by attention_heads")
    if config.layer_norm_epsilon <= 0.0:
        raise ValueError("layer_norm_epsilon must be positive")


def validate_optimization_config(config: OptimizationConfig) -> None:
    if config.optimizer not in {"sgd", "adamw"}:
        raise ValueError("optimizer must be 'sgd' or 'adamw'")
    if config.gradient_clip < 0.0:
        raise ValueError("gradient_clip must be non-negative")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= config.beta1 < 1.0:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0.0 <= config.beta2 < 1.0:
        raise ValueError("beta2 must be in [0, 1)")
    if config.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if config.warmup_steps < 0 or config.decay_steps < 0:
        raise ValueError("warmup and decay steps must be non-negative")
    if config.min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")


def validate_generation_config(config: GenerationConfig) -> None:
    if config.temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if config.top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if config.repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    if config.trace_top_tokens <= 0:
        raise ValueError("trace_top_tokens must be positive")


def transformer_config_from_args(args: Any, vocab_size: int) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=vocab_size,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        feedforward_dim=args.feedforward_dim,
        seed=args.seed,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        use_layer_norm=args.use_layer_norm,
        use_pre_layer_norm=args.use_pre_layer_norm,
        use_rms_norm=args.use_rms_norm,
        layer_norm_epsilon=args.layer_norm_epsilon,
        use_gated_mlp=args.use_gated_mlp,
        tie_output_embeddings=args.tie_output_embeddings,
        use_rotary_positions=args.use_rotary_positions,
        use_kv_cache_path=args.use_kv_cache_path,
        use_context_mean=args.use_context_mean,
        use_context_projection=args.use_context_projection,
        use_prompt_prefix_projection=args.use_prompt_prefix_projection,
        use_prompt_position_projection=args.use_prompt_position_projection,
        prompt_position_projection_scale=args.prompt_position_projection_scale,
        use_prompt_attention_summary=args.use_prompt_attention_summary,
    )


def optimization_config_from_args(args: Any) -> OptimizationConfig:
    return OptimizationConfig(
        optimizer=args.optimizer,
        gradient_clip=args.gradient_clip,
        weight_decay=args.weight_decay,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        min_learning_rate=args.min_learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def generation_config_from_args(args: Any) -> GenerationConfig:
    return GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        trace_top_tokens=args.trace_top_tokens,
        use_kv_cache=args.use_kv_cache,
    )


def closed_world_dataset_metadata(vocab_size: int) -> dict[str, Any]:
    return {
        "tokenizer": TRANSFORMER_TOKENIZER,
        "vocab_size": vocab_size,
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
    }


def transformer_run_metadata(
    args: Any,
    tokenizer: Any,
    optimizer: Any,
    run_kind: str,
    resume: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_kind": run_kind,
        "seed": args.seed,
        "config": asdict(transformer_config_from_args(args, tokenizer.vocab_size)),
        "optimizer": optimizer.summary(),
        "resume": resume,
        "dataset": closed_world_dataset_metadata(tokenizer.vocab_size),
    }


def checkpoint_header(config: TransformerConfig) -> dict[str, Any]:
    return {
        "architecture": TRANSFORMER_ARCHITECTURE,
        "checkpoint_format": TRANSFORMER_CHECKPOINT_FORMAT,
        "config": asdict(config),
    }
