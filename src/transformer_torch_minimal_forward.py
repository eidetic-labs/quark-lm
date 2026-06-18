"""Minimal PyTorch forward parity for the simplest transformer profile."""

from __future__ import annotations

import math
from typing import Any

from transformer_model import GenerationConfig
from transformer_sampling import generation_distribution


MINIMAL_TORCH_FORWARD_STATUS = "minimal_forward"


def torch_minimal_parity_outputs(
    *,
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Compute parity outputs for the default one-layer transformer profile."""

    unsupported = _unsupported_reason(fixture["model_config"])
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


def _unsupported_reason(config: dict[str, Any]) -> str | None:
    unsupported_flags = [
        "use_layer_norm",
        "use_pre_layer_norm",
        "use_rms_norm",
        "use_gated_mlp",
        "tie_output_embeddings",
        "use_rotary_positions",
        "use_kv_cache_path",
        "use_context_mean",
        "use_context_projection",
        "use_prompt_prefix_projection",
        "use_prompt_position_projection",
        "use_prompt_attention_summary",
    ]
    if config.get("num_layers") != 1:
        return "minimal PyTorch parity supports exactly one transformer layer"
    if config.get("attention_heads") != 1:
        return "minimal PyTorch parity supports exactly one attention head"
    for flag in unsupported_flags:
        if config.get(flag):
            return f"minimal PyTorch parity does not support {flag}"
    return None


def _forward_case(
    case: dict[str, Any],
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    logits = _logits(case["context"], fixture, torch, runtime)
    probs = _to_list(torch.softmax(logits, dim=0))
    loss = -math.log(max(probs[case["target"]], 1e-12))
    logits_list = _to_list(logits)
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
        probs = _to_list(torch.softmax(_logits(context, fixture, torch, runtime), dim=0))
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


def _logits(
    context: list[int],
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    weights = fixture["weights"]
    config = fixture["model_config"]
    token_embeddings = _tensor(torch, weights["token_embeddings"], runtime)
    position_embeddings = _tensor(torch, weights["position_embeddings"], runtime)
    x = torch.stack(
        [
            token_embeddings[token_id] + position_embeddings[position]
            for position, token_id in enumerate(context)
        ]
    )
    q = torch.stack([_linear(row, weights["wq"], weights["bq"], torch, runtime) for row in x])
    k = torch.stack([_linear(row, weights["wk"], weights["bk"], torch, runtime) for row in x])
    v = torch.stack([_linear(row, weights["wv"], weights["bv"], torch, runtime) for row in x])
    attended = _causal_attention(q, k, v, config, torch)
    projected = _linear(attended, weights["wo"], weights["bo"], torch, runtime)
    last_position = config["context_size"] - 1
    hidden = x[last_position] + projected
    ff_hidden = torch.tanh(_linear(hidden, weights["w1"], weights["b1"], torch, runtime))
    ff_out = _linear(ff_hidden, weights["w2"], weights["b2"], torch, runtime)
    final_hidden = hidden + ff_out
    return _linear(final_hidden, weights["wout"], weights["bout"], torch, runtime)


def _causal_attention(
    q: Any,
    k: Any,
    v: Any,
    config: dict[str, Any],
    torch: Any,
) -> Any:
    position = config["context_size"] - 1
    head_dim = config["embedding_dim"]
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.stack(
        [
            (q[position] * k[past]).sum() * scale
            for past in range(position + 1)
        ]
    )
    weights = torch.softmax(scores, dim=0)
    return torch.stack(
        [
            torch.stack(
                [weights[past] * v[past][dim] for past in range(position + 1)]
            ).sum()
            for dim in range(head_dim)
        ]
    )


def _linear(
    inputs: Any,
    weights: list[list[float]],
    bias: list[float],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    return (inputs @ _tensor(torch, weights, runtime)) + _tensor(torch, bias, runtime)


def _tensor(torch: Any, value: Any, runtime: dict[str, Any]) -> Any:
    return torch.tensor(
        value,
        dtype=getattr(torch, runtime["dtype"]),
        device=runtime["device"],
    )


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return [float(item) for item in value.tolist()]


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
