"""Optimizer-step control execution for optional PyTorch training parity."""

from __future__ import annotations

from typing import Any

from transformer_torch_adamw_expected_update import (
    TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS,
    build_torch_adamw_expected_update,
)
from transformer_torch_gradient_clip import apply_torch_gradient_value_clip
from transformer_torch_gradient_accumulation import (
    build_torch_gradient_accumulation_report,
)
from transformer_torch_optimizer_step_probe import (
    TORCH_OPTIMIZER_STEP_READY_STATUS,
)
from transformer_torch_optimizer_step_control import (
    TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS,
    execute_torch_optimizer_step_records,
    final_torch_optimizer_state,
    torch_optimizer_step_control_status,
    torch_step_records_match_contract,
)
from transformer_torch_parameter_mutation import (
    build_torch_parameter_mutation_report,
    snapshot_torch_parameters,
)
from transformer_torch_parameter_signature_comparison import (
    build_torch_parameter_signature_comparison,
)


TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION = 1


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
    gradient_accumulation = build_torch_gradient_accumulation_report(
        state=state,
        contract=contract,
    )
    parameters_before = snapshot_torch_parameters(state)
    expected_adamw_update = build_torch_adamw_expected_update(
        state=state,
        parameters_before=parameters_before,
        contract=contract,
    )
    step_records = execute_torch_optimizer_step_records(
        optimizer=optimizer,
        contract=contract,
    )
    parameter_mutation = build_torch_parameter_mutation_report(
        before=parameters_before,
        after=snapshot_torch_parameters(state),
    )
    signature_comparison = build_torch_parameter_signature_comparison(
        expected_signature=fixture["training_case"]["parameter_signature"],
        actual_signature=parameter_mutation["after_signature"],
        tolerance=fixture["tolerance"],
    )
    adamw_update_signature_comparison = _adamw_update_signature_comparison(
        expected_adamw_update=expected_adamw_update,
        actual_signature=parameter_mutation["after_signature"],
        tolerance=fixture["tolerance"],
    )
    final_state = final_torch_optimizer_state(
        contract=contract,
        step_records=step_records,
    )
    applied_update_count = sum(
        record["optimizer_step_called"] for record in step_records
    )
    return {
        "schema_version": TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
        "status": _status(contract, step_records, final_state),
        "reason": "optimizer control flow matches scalar step contract",
        "optimizer": contract["optimizer"],
        "step_records": step_records,
        "optimizer_state": final_state,
        "step_records_match_contract": torch_step_records_match_contract(
            contract=contract,
            step_records=step_records,
        ),
        "final_state_matches_contract": final_state
        == contract["expected_final_optimizer_state"],
        "parameter_count": contract["parameter_count"],
        "applied_update_count": applied_update_count,
        "gradient_clip": gradient_clip,
        "gradient_accumulation": gradient_accumulation,
        "parameter_mutation": parameter_mutation,
        "expected_adamw_update": expected_adamw_update,
        "parameter_signature_comparison": signature_comparison,
        "adamw_update_signature_comparison": adamw_update_signature_comparison,
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


def _status(
    contract: dict[str, Any],
    step_records: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> str:
    return torch_optimizer_step_control_status(
        contract=contract,
        step_records=step_records,
        final_state=final_state,
    )


def _adamw_update_signature_comparison(
    *,
    expected_adamw_update: dict[str, Any],
    actual_signature: dict[str, Any],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    if expected_adamw_update["status"] != TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS:
        return {
            "status": "not_run",
            "reason": "expected AdamW update was not built",
        }
    return build_torch_parameter_signature_comparison(
        expected_signature=expected_adamw_update["expected_signature"],
        actual_signature=actual_signature,
        tolerance=tolerance,
    )


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
        "status": "not_run",
        "reason": reason,
    }
