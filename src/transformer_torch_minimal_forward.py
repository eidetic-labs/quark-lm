"""Minimal PyTorch forward parity for the simplest transformer profile."""

from __future__ import annotations

import math
from typing import Any

from neural_char_ops import make_context_positioned
from transformer_model import GenerationConfig
from transformer_sampling import generation_distribution
from transformer_torch_attention import TorchLayer0KVCache
from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_profile_support import minimal_forward_unsupported_reason
from transformer_torch_tensor_ops import torch_to_list


MINIMAL_TORCH_FORWARD_STATUS = "minimal_forward"


def torch_minimal_parity_outputs(
    *,
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Compute parity outputs for the default one-layer transformer profile."""

    unsupported = minimal_forward_unsupported_reason(fixture["model_config"])
    if unsupported is not None:
        return {
            "implementation_status": "unsupported_profile",
            "parity_status": "pending",
            "forward_cases": [
                _unsupported_case(case["case_id"], unsupported)
                for case in fixture["forward_cases"]
            ],
            "generation_cases": [
                _unsupported_case(case["case_id"], unsupported)
                for case in fixture.get("generation_cases", [])
            ],
        }

    forward_cases = [
        _forward_case(case, fixture, torch, runtime)
        for case in fixture["forward_cases"]
    ]
    generation_cases = [
        _generation_case(case, fixture, torch, runtime)
        for case in fixture.get("generation_cases", [])
    ]
    return {
        "implementation_status": MINIMAL_TORCH_FORWARD_STATUS,
        "parity_status": "matched",
        "forward_cases": forward_cases,
        "generation_cases": generation_cases,
    }


def _forward_case(
    case: dict[str, Any],
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    logits = torch_minimal_logits(case["context"], fixture, torch, runtime)
    probs = torch_to_list(torch.softmax(logits, dim=0))
    loss = -math.log(max(probs[case["target"]], 1e-12))
    logits_list = torch_to_list(logits)
    return {
        "case_id": case["case_id"],
        "status": "computed",
        "context": list(case["context"]),
        "target": case["target"],
        "logits": logits_list,
        "probabilities": probs,
        "loss": loss,
        "predicted_token_id": max(
            range(len(logits_list)),
            key=lambda index: logits_list[index],
        ),
    }


def _generation_case(
    case: dict[str, Any],
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    config = GenerationConfig(**case["generation_config"])
    if config.temperature > 0.0:
        return _unsupported_case(
            case["case_id"],
            "minimal PyTorch parity supports greedy generation only",
        )
    ids = list(case["prompt_ids"])
    generated: list[int] = []
    cache_enabled = config.use_kv_cache or fixture["model_config"].get(
        "use_kv_cache_path",
        False,
    )
    cache_events: list[dict[str, Any]] = []
    # Per-step runtime carries abs_positions; copy so the caller's runtime dict stays
    # pristine (the runtime-report parity check compares it by equality).
    step_runtime = dict(runtime)
    # Phase-3 layer-0-only append-valid KV cache (torch mirror), threaded via the
    # runtime; a fresh instance per generation, None when the path is disabled so the
    # forward is byte-identical to cache-off. The block consults it for layer 0 only.
    kv_cache = TorchLayer0KVCache() if cache_enabled else None
    if kv_cache is not None:
        step_runtime["kv_cache"] = kv_cache
    for _step in range(case["max_new_chars"]):
        # Single source of truth for the window + each slot's ABSOLUTE stream index
        # (shared with the scalar generation boundary; removes the duplicate window
        # builder's drift risk). The window content is byte-identical to the old
        # _make_context, so flag-off stays byte-exact; runtime['abs_positions'] is
        # consumed at the RoPE site only when use_absolute_rope is on.
        context, abs_positions = make_context_positioned(
            ids,
            fixture["model_config"]["context_size"],
            fixture["tokenizer"]["pad_id"],
        )
        assert len(abs_positions) == fixture["model_config"]["context_size"]
        step_runtime["abs_positions"] = abs_positions
        if cache_enabled:
            cache_events.append(
                {
                    "context_length": len(context),
                    "source_token_count": len(ids),
                    "sliding_window": (
                        len(ids) > fixture["model_config"]["context_size"]
                    ),
                }
            )
        probs = torch_to_list(
            torch.softmax(
                torch_minimal_logits(context, fixture, torch, step_runtime), dim=0
            )
        )
        filtered = generation_distribution(probs, generated, config)
        next_id = max(range(len(filtered)), key=lambda index: filtered[index])
        ids.append(next_id)
        generated.append(next_id)
    return {
        "case_id": case["case_id"],
        "status": "computed",
        "text": _decode(generated, fixture["tokenizer"]),
        "token_ids": generated,
        "cache": {
            "enabled": cache_enabled,
            "mode": "rolling-context-kv-aware" if cache_enabled else "disabled",
            "events": cache_events,
            "hits": kv_cache.hits if kv_cache is not None else 0,
            "writes": kv_cache.writes if kv_cache is not None else 0,
            "pad_skips": kv_cache.pad_skips if kv_cache is not None else 0,
        },
    }


def _decode(token_ids: list[int], tokenizer: dict[str, Any]) -> str:
    tokens = tokenizer.get("tokens", [])
    pad_id = tokenizer["pad_id"]
    return "".join(tokens[token_id] for token_id in token_ids if token_id != pad_id)


def _unsupported_case(case_id: str, reason: str) -> dict[str, Any]:
    return {"case_id": case_id, "status": "pending", "reason": reason}
