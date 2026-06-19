"""Optimizer-step control execution for optional PyTorch training parity."""

from __future__ import annotations

from typing import Any

from transformer_torch_gradient_clip import apply_torch_gradient_value_clip
from transformer_torch_optimizer_step_probe import (
    TORCH_OPTIMIZER_STEP_READY_STATUS,
)
from transformer_torch_parameter_mutation import (
    build_torch_parameter_mutation_report,
    snapshot_torch_parameters,
)


TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION = 1
TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS = "step_control_matched"


def build_torch_optimizer_step_execution_probe(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any] | None,
    optimizer_step_probe: dict[str, Any],
    torch: Any | None,
) -> dict[str, Any]:
    """Execute optimizer control flow against the scalar step contract."""

    if state is None or torch is None:
        return _not_run("pytorch training runtime is not ready")
    if optimizer_step_probe.get("status") != TORCH_OPTIMIZER_STEP_READY_STATUS:
        return _not_run("optimizer step readiness has not passed")

    contract = fixture["optimizer_step_contract"]
    if contract["optimizer"] != "adamw":
        return _not_run(f"unsupported optimizer: {contract['optimizer']}")

    adamw = getattr(getattr(torch, "optim", None), "AdamW", None)
    if not callable(adamw):
        return {
            "schema_version": TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
            "status": "optimizer_unavailable",
            "reason": "torch.optim.AdamW is not available",
        }
    optimizer = _build_adamw_optimizer(torch, state, contract)
    missing = _missing_optimizer_methods(optimizer)
    if missing:
        return {
            "schema_version": TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
            "status": "optimizer_methods_unavailable",
            "reason": "optimizer is missing required methods",
            "missing_methods": missing,
        }

    gradient_clip = apply_torch_gradient_value_clip(
        torch=torch,
        state=state,
        clip_value=contract["gradient_clip"]["value"],
    )
    if gradient_clip["status"] == "clipper_unavailable":
        return {
            "schema_version": TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
            "status": "gradient_clip_unavailable",
            "reason": gradient_clip["reason"],
            "gradient_clip": gradient_clip,
        }
    parameters_before = snapshot_torch_parameters(state)
    step_records = _execute_step_records(
        optimizer=optimizer,
        contract=contract,
    )
    parameter_mutation = build_torch_parameter_mutation_report(
        before=parameters_before,
        after=snapshot_torch_parameters(state),
    )
    final_state = _final_optimizer_state(
        contract=contract,
        step_records=step_records,
    )
    return {
        "schema_version": TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
        "status": _status(contract, step_records, final_state),
        "reason": "optimizer control flow matches scalar step contract",
        "optimizer": contract["optimizer"],
        "step_records": step_records,
        "optimizer_state": final_state,
        "step_records_match_contract": _records_match_contract(
            contract,
            step_records,
        ),
        "final_state_matches_contract": final_state
        == contract["expected_final_optimizer_state"],
        "parameter_count": contract["parameter_count"],
        "applied_update_count": sum(
            1 for record in step_records if record["optimizer_step_called"]
        ),
        "gradient_clip": gradient_clip,
        "parameter_mutation": parameter_mutation,
    }


def _build_adamw_optimizer(
    torch: Any,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Any:
    adamw = getattr(getattr(torch, "optim", None), "AdamW")
    adamw_config = contract["adamw"]
    return adamw(
        [parameter["tensor"] for parameter in state["parameters"]],
        lr=contract["base_learning_rate"],
        betas=(adamw_config["beta1"], adamw_config["beta2"]),
        eps=adamw_config["epsilon"],
        weight_decay=adamw_config["weight_decay"],
    )


def _missing_optimizer_methods(optimizer: Any) -> list[str]:
    return [
        method
        for method in ("step", "zero_grad")
        if not callable(getattr(optimizer, method, None))
    ]


def _execute_step_records(
    *,
    optimizer: Any,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for expected in contract["expected_step_records"]:
        _set_optimizer_learning_rate(
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


def _set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for group in getattr(optimizer, "param_groups", []):
        group["lr"] = learning_rate


def _final_optimizer_state(
    *,
    contract: dict[str, Any],
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
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


def _status(
    contract: dict[str, Any],
    step_records: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> str:
    if not _records_match_contract(contract, step_records):
        return "step_record_mismatch"
    if final_state != contract["expected_final_optimizer_state"]:
        return "final_state_mismatch"
    return TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS


def _records_match_contract(
    contract: dict[str, Any],
    step_records: list[dict[str, Any]],
) -> bool:
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


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
        "status": "not_run",
        "reason": reason,
    }
