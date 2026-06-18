"""Minimal PyTorch forward parity for the simplest transformer profile."""

from __future__ import annotations

import math
from typing import Any

from transformer_model import GenerationConfig
from transformer_sampling import generation_distribution
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
    for _step in range(case["max_new_chars"]):
        context = _make_context(
            ids,
            fixture["model_config"]["context_size"],
            fixture["tokenizer"]["pad_id"],
        )
        probs = torch_to_list(
            torch.softmax(torch_minimal_logits(context, fixture, torch, runtime), dim=0)
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
    }


def _make_context(ids: list[int], context_size: int, pad_id: int) -> list[int]:
    if len(ids) >= context_size:
        return ids[-context_size:]
    return [pad_id] * (context_size - len(ids)) + ids


def _decode(token_ids: list[int], tokenizer: dict[str, Any]) -> str:
    tokens = tokenizer.get("tokens", [])
    pad_id = tokenizer["pad_id"]
    return "".join(tokens[token_id] for token_id in token_ids if token_id != pad_id)


def _unsupported_case(case_id: str, reason: str) -> dict[str, Any]:
    return {"case_id": case_id, "status": "pending", "reason": reason}
