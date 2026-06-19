"""Checkpoint compatibility checks for replay-updated PyTorch states."""

from __future__ import annotations

import math
from typing import Any

from transformer_checkpoint import checkpoint_summary, validate_checkpoint_payload
from transformer_model import TransformerConfig, checkpoint_header
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_replay_final_evaluation import (
    TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
)
from transformer_torch_replay_update_state import (
    TORCH_REPLAY_UPDATE_STATE_APPLIED_STATUS,
    apply_torch_replay_update_to_state,
)
from transformer_torch_training_state import torch_training_weights_from_state


TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION = 1
TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS = "replay_checkpoint_compatible"
TORCH_REPLAY_CHECKPOINT_MISMATCH_STATUS = "replay_checkpoint_mismatch"
TORCH_REPLAY_CHECKPOINT_NOT_RUN_STATUS = "replay_checkpoint_not_run"


def build_torch_replay_checkpoint_compatibility(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any] | None,
    torch: Any | None,
    runtime: dict[str, Any],
    replay_control_probe: dict[str, Any],
    buffer_comparison: dict[str, Any],
    final_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Validate that replay-updated tensors can round-trip as a checkpoint."""

    if final_evaluation.get("status") != TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS:
        return _not_run("replay final evaluation has not passed")
    update = apply_torch_replay_update_to_state(
        fixture=fixture,
        state=state,
        torch=torch,
        runtime=runtime,
        replay_control_probe=replay_control_probe,
        buffer_comparison=buffer_comparison,
    )
    if update["status"] != TORCH_REPLAY_UPDATE_STATE_APPLIED_STATUS:
        return _not_run(update["reason"])

    payload = _checkpoint_payload(fixture=fixture, state=state)
    validate_checkpoint_payload(payload)
    model, tokenizer = TinyTransformerLM.from_dict(payload)
    case = fixture["training_case"]
    logits = model._forward_floats(case["context"])
    loss = model.nll(case["context"], case["target"])
    logits_match = _logits_match(fixture=fixture, actual=logits)
    loss_match = math.isclose(
        case["final_loss"],
        loss,
        abs_tol=fixture["tolerance"]["absolute"],
        rel_tol=fixture["tolerance"]["relative"],
    )
    passed = logits_match and loss_match and tokenizer is not None
    return {
        "schema_version": TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
        "status": (
            TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS
            if passed
            else TORCH_REPLAY_CHECKPOINT_MISMATCH_STATUS
        ),
        "passed": passed,
        "case_id": case["case_id"],
        "checkpoint_summary": checkpoint_summary(payload),
        "round_trip_loaded": True,
        "tokenizer_round_trip": tokenizer is not None,
        "loss_abs_diff": abs(case["final_loss"] - loss),
        "max_logit_abs_diff": _max_abs_diff(case["final_logits"], logits),
        "checkpoint_parity_proven": passed,
        "promoted_training_backend": False,
        "reason": _reason(passed),
    }


def _checkpoint_payload(*, fixture: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        **checkpoint_header(TransformerConfig(**fixture["model_config"])),
        "weights": _json_safe(
            torch_training_weights_from_state(fixture=fixture, state=state)
        ),
        "tokenizer": dict(fixture["tokenizer"]),
        "metadata": {
            "backend": "pytorch_replay_candidate",
            "parity_source": "replay_checkpoint_compatibility",
            "promoted": False,
        },
    }


def _json_safe(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, int | float):
        return float(value)
    return value


def _logits_match(*, fixture: dict[str, Any], actual: list[float]) -> bool:
    tolerance = fixture["tolerance"]
    return all(
        math.isclose(
            expected,
            actual_value,
            abs_tol=tolerance["absolute"],
            rel_tol=tolerance["relative"],
        )
        for expected, actual_value in zip(
            fixture["training_case"]["final_logits"],
            actual,
        )
    )


def _max_abs_diff(expected: list[float], actual: list[float]) -> float:
    return max(
        (abs(left - right) for left, right in zip(expected, actual)),
        default=0.0,
    )


def _reason(passed: bool) -> str:
    return (
        "replay-updated tensors round-trip through the checkpoint format"
        if passed
        else "replay-updated checkpoint round-trip does not match scalar evidence"
    )


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
        "status": TORCH_REPLAY_CHECKPOINT_NOT_RUN_STATUS,
        "passed": False,
        "reason": reason,
        "checkpoint_parity_proven": False,
        "promoted_training_backend": False,
    }
