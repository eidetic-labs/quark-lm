"""Final-logit and final-loss checks after a replayed PyTorch update."""

from __future__ import annotations

import math
from typing import Any

from transformer_torch_replay_update_comparison import (
    TORCH_REPLAY_UPDATE_MATCHED_STATUS,
)
from transformer_torch_replay_update_state import (
    TORCH_REPLAY_UPDATE_STATE_APPLIED_STATUS,
    apply_torch_replay_update_to_state,
)
from transformer_torch_tensor_ops import torch_to_float, torch_to_list
from transformer_torch_training_loss import (
    build_torch_training_logits,
    build_torch_training_loss_tensor,
)


TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION = 1
TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS = "replay_final_evaluation_matched"
TORCH_REPLAY_FINAL_EVAL_MISMATCH_STATUS = "replay_final_evaluation_mismatch"
TORCH_REPLAY_FINAL_EVAL_NOT_RUN_STATUS = "replay_final_evaluation_not_run"


def build_torch_replay_final_evaluation(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any] | None,
    torch: Any | None,
    runtime: dict[str, Any],
    replay_control_probe: dict[str, Any],
    buffer_comparison: dict[str, Any],
    update_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Compare final logits and loss after replaying a verified update."""

    if update_comparison.get("status") != TORCH_REPLAY_UPDATE_MATCHED_STATUS:
        return _not_run("replay update parity has not passed")
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

    case = fixture["training_case"]
    logits = torch_to_list(
        build_torch_training_logits(
            fixture=fixture,
            state=state,
            torch=torch,
            runtime=runtime,
            context=case["context"],
        )
    )
    loss = torch_to_float(
        build_torch_training_loss_tensor(
            fixture=fixture,
            state=state,
            torch=torch,
            runtime=runtime,
            context=case["context"],
            target=case["target"],
        )
    )
    logits_match = _logits_match(fixture=fixture, actual=logits)
    loss_match = math.isclose(
        case["final_loss"],
        loss,
        abs_tol=fixture["tolerance"]["absolute"],
        rel_tol=fixture["tolerance"]["relative"],
    )
    passed = logits_match and loss_match
    return {
        "schema_version": TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
        "status": (
            TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS
            if passed
            else TORCH_REPLAY_FINAL_EVAL_MISMATCH_STATUS
        ),
        "passed": passed,
        "case_id": case["case_id"],
        "optimizer_update_status": update_comparison["status"],
        "final_logits": logits,
        "final_loss": loss,
        "loss_abs_diff": abs(case["final_loss"] - loss),
        "max_logit_abs_diff": _max_abs_diff(case["final_logits"], logits),
        "final_logit_parity_proven": logits_match,
        "final_loss_parity_proven": loss_match,
        "checkpoint_parity_proven": False,
        "reason": _reason(passed),
    }


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
        "replayed update final logits and loss match scalar evidence"
        if passed
        else "replayed update final logits or loss do not match scalar evidence"
    )


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
        "status": TORCH_REPLAY_FINAL_EVAL_NOT_RUN_STATUS,
        "passed": False,
        "reason": reason,
        "final_logit_parity_proven": False,
        "final_loss_parity_proven": False,
        "checkpoint_parity_proven": False,
    }
