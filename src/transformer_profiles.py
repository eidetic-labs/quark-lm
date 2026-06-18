"""Named transformer profiles for controlled architecture experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Any


DEFAULT_PROFILE = "default"
MODERN_SMALL_PROFILE = "modern_small"
TRANSFORMER_PROFILES = {DEFAULT_PROFILE, MODERN_SMALL_PROFILE}


def apply_transformer_profile(
    config: Any,
    profile: str,
) -> Any:
    if profile == DEFAULT_PROFILE:
        return config
    if profile != MODERN_SMALL_PROFILE:
        raise ValueError(f"unsupported transformer profile: {profile!r}")
    heads = 2 if config.embedding_dim % 2 == 0 else 1
    return replace(
        config,
        attention_heads=max(config.attention_heads, heads),
        use_pre_layer_norm=True,
        use_rms_norm=True,
        use_gated_mlp=True,
        use_rotary_positions=True,
    )


def apply_optimizer_profile(
    config: Any,
    profile: str,
) -> Any:
    if profile == DEFAULT_PROFILE:
        return config
    if profile != MODERN_SMALL_PROFILE:
        raise ValueError(f"unsupported transformer profile: {profile!r}")
    return replace(
        config,
        optimizer="adamw",
        gradient_clip=min(config.gradient_clip, 2.0),
        warmup_steps=max(config.warmup_steps, 5),
        decay_steps=max(config.decay_steps, 20),
        gradient_accumulation_steps=max(config.gradient_accumulation_steps, 2),
    )
