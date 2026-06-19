"""Compare replay-buffer AdamW updates with scalar training evidence."""

from __future__ import annotations

from typing import Any

from transformer_torch_parameter_mutation import (
    build_torch_parameter_mutation_report,
)
from transformer_torch_parameter_signature_comparison import (
    build_torch_parameter_signature_comparison,
)
from transformer_torch_replay_update_state import (
    TORCH_REPLAY_UPDATE_STATE_APPLIED_STATUS,
    apply_torch_replay_update_to_state,
)


TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION = 1
TORCH_REPLAY_UPDATE_MATCHED_STATUS = "replay_update_signature_matched"
TORCH_REPLAY_UPDATE_MISMATCH_STATUS = "replay_update_signature_mismatch"
TORCH_REPLAY_UPDATE_NOT_RUN_STATUS = "replay_update_comparison_not_run"


def build_torch_replay_update_comparison(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any] | None,
    torch: Any | None,
    runtime: dict[str, Any],
    replay_control_probe: dict[str, Any],
    buffer_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Apply matched replay-buffer gradients and compare the update result."""

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

    signature_comparison = build_torch_parameter_signature_comparison(
        expected_signature=fixture["training_case"]["trainable_parameter_signature"],
        actual_signature=update["parameters_after"]["signature"],
        tolerance=fixture["tolerance"],
    )
    return {
        "schema_version": TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
        "status": (
            TORCH_REPLAY_UPDATE_MATCHED_STATUS
            if signature_comparison["passed"]
            else TORCH_REPLAY_UPDATE_MISMATCH_STATUS
        ),
        "passed": signature_comparison["passed"],
        "case_id": replay_control_probe["case_id"],
        "optimizer": "adamw",
        "applied_update_count": 1,
        "effective_learning_rate": update["effective_learning_rate"],
        "compared_signature": "scalar_trainable_parameter_signature",
        "gradient_signature": update["gradient_signature"],
        "parameter_mutation": build_torch_parameter_mutation_report(
            before=update["parameters_before"],
            after=update["parameters_after"],
        ),
        "parameter_signature_comparison": signature_comparison,
        "optimizer_update_parity_proven": signature_comparison["passed"],
        "final_logit_parity_proven": False,
        "final_loss_parity_proven": False,
        "reason": _reason(signature_comparison["passed"]),
    }


def _reason(passed: bool) -> str:
    return (
        "replayed AdamW update matches scalar final parameter signature"
        if passed
        else "replayed AdamW update does not match scalar final parameter signature"
    )


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
        "status": TORCH_REPLAY_UPDATE_NOT_RUN_STATUS,
        "passed": False,
        "reason": reason,
        "optimizer_update_parity_proven": False,
        "final_logit_parity_proven": False,
        "final_loss_parity_proven": False,
    }
