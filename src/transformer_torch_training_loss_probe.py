"""Initial-loss probes for optional PyTorch training parity."""

from __future__ import annotations

import math
from typing import Any

from transformer_torch_tensor_ops import torch_to_list
from transformer_torch_training_loss import build_torch_training_logits


TORCH_TRAINING_LOSS_PROBE_SCHEMA_VERSION = 1


def build_torch_training_initial_loss_probe(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Compute initial logits and loss from trainable tensors."""

    case = fixture["training_case"]
    logits = build_torch_training_logits(
        fixture=fixture,
        state=state,
        torch=torch,
        runtime=runtime,
        context=case["context"],
    )
    logits_list = torch_to_list(logits)
    probabilities = torch_to_list(torch.softmax(logits, dim=0))
    loss = -math.log(max(probabilities[case["target"]], 1e-12))
    return {
        "schema_version": TORCH_TRAINING_LOSS_PROBE_SCHEMA_VERSION,
        "status": (
            "matched"
            if _matches_fixture(fixture, logits_list, loss)
            else "drifted"
        ),
        "case_id": case["case_id"],
        "initial_loss": loss,
        "initial_logits": logits_list,
        "loss_abs_diff": abs(case["initial_loss"] - loss),
        "max_logit_abs_diff": _max_abs_diff(case["initial_logits"], logits_list),
    }


def _matches_fixture(
    fixture: dict[str, Any],
    logits: list[float],
    loss: float,
) -> bool:
    tolerance = fixture["tolerance"]
    case = fixture["training_case"]
    return math.isclose(
        case["initial_loss"],
        loss,
        abs_tol=tolerance["absolute"],
        rel_tol=tolerance["relative"],
    ) and all(
        math.isclose(
            expected,
            actual,
            abs_tol=tolerance["absolute"],
            rel_tol=tolerance["relative"],
        )
        for expected, actual in zip(case["initial_logits"], logits)
    )


def _max_abs_diff(expected: list[float], actual: list[float]) -> float:
    return max(
        (abs(left - right) for left, right in zip(expected, actual)),
        default=0.0,
    )
