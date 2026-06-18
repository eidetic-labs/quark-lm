"""Build scalar-reference fixtures for transformer backend parity."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from transformer_backend_parity_schema import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    PARITY_FIXTURE_KIND,
    PARITY_SCHEMA_VERSION,
)
from transformer_backend_parity_validation import validate_backend_parity_fixture
from transformer_backend_policy import SCALAR_BACKEND, transformer_backend_metadata
from transformer_math import cross_entropy_scalars, softmax_floats
from transformer_model import (
    TRANSFORMER_ARCHITECTURE,
    GenerationConfig,
    validate_generation_config,
)


def build_scalar_backend_parity_fixture(
    *,
    fixture_id: str,
    model: Any,
    tokenizer: Any,
    contexts: list[list[int]],
    targets: list[int],
    prompts: list[str],
    corpus_hash: str,
    generation_config: GenerationConfig | None = None,
    max_new_chars: int = 3,
    tokenizer_manifest_hash: str | None = None,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Capture scalar behavior a future backend must match."""

    if not fixture_id:
        raise ValueError("fixture_id is required")
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have the same length")
    config = generation_config or GenerationConfig()
    validate_generation_config(config)
    fixture = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "kind": PARITY_FIXTURE_KIND,
        "fixture_id": fixture_id,
        "architecture": TRANSFORMER_ARCHITECTURE,
        "reference_backend": transformer_backend_metadata(
            active_backend=SCALAR_BACKEND,
            seed=model.config.seed,
            tokenizer_type=getattr(tokenizer, "tokenizer_type", "char"),
            corpus_hash=corpus_hash,
            tokenizer_manifest_hash=tokenizer_manifest_hash,
        ),
        "model_config": asdict(model.config),
        "tokenizer": _tokenizer_summary(tokenizer, tokenizer_manifest_hash),
        "tolerance": {"absolute": absolute_tolerance, "relative": relative_tolerance},
        "forward_cases": [
            _forward_case(model, context, target, index)
            for index, (context, target) in enumerate(zip(contexts, targets), start=1)
        ],
        "generation_cases": [
            _generation_case(model, tokenizer, prompt, index, config, max_new_chars)
            for index, prompt in enumerate(prompts, start=1)
        ],
    }
    validate_backend_parity_fixture(fixture)
    return fixture


def _tokenizer_summary(tokenizer: Any, manifest_hash: str | None) -> dict[str, Any]:
    return {
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "tokenizer_manifest_hash": manifest_hash,
    }


def _forward_case(
    model: Any,
    context: list[int],
    target: int,
    index: int,
) -> dict[str, Any]:
    scalar_logits = model._forward_scalars(context)
    logits = [value.data for value in scalar_logits]
    loss = cross_entropy_scalars(scalar_logits, target).data
    probabilities = softmax_floats(logits)
    return {
        "case_id": f"forward-{index:02d}",
        "context": list(context),
        "target": target,
        "logits": logits,
        "probabilities": probabilities,
        "loss": loss,
        "predicted_token_id": max(range(len(logits)), key=lambda item: logits[item]),
    }


def _generation_case(
    model: Any,
    tokenizer: Any,
    prompt: str,
    index: int,
    config: GenerationConfig,
    max_new_chars: int,
) -> dict[str, Any]:
    result = model.generate_with_trace(
        tokenizer,
        prompt,
        max_new_chars,
        config,
    )
    return {
        "case_id": f"generation-{index:02d}",
        "prompt": prompt,
        "prompt_ids": tokenizer.encode(prompt),
        "max_new_chars": max_new_chars,
        "generation_config": asdict(config),
        "text": result["text"],
        "token_ids": [step["token_id"] for step in result["trace"]],
        "trace": result["trace"],
    }
