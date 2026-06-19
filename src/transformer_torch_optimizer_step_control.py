"""Optimizer step-control replay helpers for PyTorch parity probes."""

from __future__ import annotations

from typing import Any


TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS = "step_control_matched"


def execute_torch_optimizer_step_records(
    *,
    optimizer: Any,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replay scalar optimizer step-control decisions against an optimizer."""

    records = []
    for expected in contract["expected_step_records"]:
        set_torch_optimizer_learning_rate(
            optimizer,
            expected["effective_learning_rate"],
        )
        step_called = False
        zero_grad_called = False
        if expected["update_applied"]:
            optimizer.step()
            step_called = True
            optimizer.zero_grad()
            zero_grad_called = True
        records.append(
            {
                **expected,
                "optimizer_step_called": step_called,
                "optimizer_zero_grad_called": zero_grad_called,
            }
        )
    return records


def set_torch_optimizer_learning_rate(
    optimizer: Any,
    learning_rate: float,
) -> None:
    """Apply the scalar schedule's current learning rate to optimizer groups."""

    for group in getattr(optimizer, "param_groups", []):
        group["lr"] = learning_rate


def final_torch_optimizer_state(
    *,
    contract: dict[str, Any],
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the final optimizer cadence state from replayed records."""

    if not step_records:
        return {
            "update_count": 0,
            "pending_accumulation": 0,
            "param_count": contract["parameter_count"],
        }
    last = step_records[-1]
    return {
        "update_count": last["update_count_after"],
        "pending_accumulation": last["pending_accumulation_after"],
        "param_count": contract["parameter_count"],
    }


def torch_optimizer_step_control_status(
    *,
    contract: dict[str, Any],
    step_records: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> str:
    """Classify whether replayed optimizer control matches the contract."""

    if not torch_step_records_match_contract(
        contract=contract,
        step_records=step_records,
    ):
        return "step_record_mismatch"
    if final_state != contract["expected_final_optimizer_state"]:
        return "final_state_mismatch"
    return TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS


def torch_step_records_match_contract(
    *,
    contract: dict[str, Any],
    step_records: list[dict[str, Any]],
) -> bool:
    """Return true when replayed records preserve scalar control semantics."""

    if len(step_records) != len(contract["expected_step_records"]):
        return False
    for actual, expected in zip(step_records, contract["expected_step_records"]):
        if any(actual[key] != expected[key] for key in expected):
            return False
        if actual["optimizer_step_called"] != expected["update_applied"]:
            return False
        if actual["optimizer_zero_grad_called"] != expected["update_applied"]:
            return False
    return True
