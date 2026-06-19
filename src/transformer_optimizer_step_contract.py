"""Scalar optimizer-step contracts for training parity artifacts."""

from __future__ import annotations

from typing import Any


OPTIMIZER_STEP_CONTRACT_SCHEMA_VERSION = 1
OPTIMIZER_STEP_CONTRACT_KIND = "transformer_optimizer_step_contract"


def build_optimizer_step_contract(
    *,
    optimizer_config: dict[str, Any],
    parameter_manifest: dict[str, Any],
    training_case: dict[str, Any],
) -> dict[str, Any]:
    """Build the scalar optimizer behavior contract a backend must match."""

    records = _expected_step_records(
        optimizer_config=optimizer_config,
        base_learning_rate=training_case["learning_rate"],
        steps=training_case["steps"],
    )
    contract = {
        "schema_version": OPTIMIZER_STEP_CONTRACT_SCHEMA_VERSION,
        "kind": OPTIMIZER_STEP_CONTRACT_KIND,
        "optimizer": optimizer_config["optimizer"],
        "parameter_order": parameter_manifest["parameter_order"],
        "parameter_count": parameter_manifest["parameter_count"],
        "gradient_source": "tensor.grad",
        "gradient_clip": {
            "mode": "per_parameter_value",
            "value": optimizer_config["gradient_clip"],
        },
        "gradient_accumulation_steps": optimizer_config[
            "gradient_accumulation_steps"
        ],
        "base_learning_rate": training_case["learning_rate"],
        "schedule": {
            "warmup_steps": optimizer_config["warmup_steps"],
            "decay_steps": optimizer_config["decay_steps"],
            "min_learning_rate": optimizer_config["min_learning_rate"],
        },
        "adamw": _adamw_contract(optimizer_config),
        "expected_step_records": records,
        "expected_final_optimizer_state": _expected_final_optimizer_state(
            parameter_manifest=parameter_manifest,
            records=records,
        ),
    }
    validate_optimizer_step_contract(contract, training_case=training_case)
    return contract


def validate_optimizer_step_contract(
    contract: dict[str, Any],
    *,
    training_case: dict[str, Any],
) -> None:
    """Validate an optimizer-step contract against scalar fixture evidence."""

    if contract.get("schema_version") != OPTIMIZER_STEP_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer step contract schema_version")
    if contract.get("kind") != OPTIMIZER_STEP_CONTRACT_KIND:
        raise ValueError(f"kind must be {OPTIMIZER_STEP_CONTRACT_KIND}")
    records = contract.get("expected_step_records")
    if not isinstance(records, list) or len(records) != training_case["steps"]:
        raise ValueError("optimizer step contract records do not match step count")
    for expected, actual in zip(records, training_case["step_records"]):
        _validate_step_record(expected, actual)
    final_state = training_case["optimizer_state"]
    expected_final = contract.get("expected_final_optimizer_state", {})
    for key in ("update_count", "pending_accumulation", "param_count"):
        if expected_final.get(key) != final_state.get(key):
            raise ValueError(f"optimizer step contract final {key} mismatch")


def _expected_step_records(
    *,
    optimizer_config: dict[str, Any],
    base_learning_rate: float,
    steps: int,
) -> list[dict[str, Any]]:
    records = []
    update_count = 0
    pending_accumulation = 0
    accumulation_steps = optimizer_config["gradient_accumulation_steps"]
    for step in range(1, steps + 1):
        update_before = update_count
        pending_before = pending_accumulation
        pending_accumulation += 1
        update_applied = pending_accumulation >= accumulation_steps
        if update_applied:
            pending_accumulation = 0
            update_count += 1
            lr_step = update_count
        else:
            lr_step = update_count + 1
        records.append(
            {
                "step": step,
                "update_applied": update_applied,
                "update_count_before": update_before,
                "update_count_after": update_count,
                "pending_accumulation_before": pending_before,
                "pending_accumulation_after": pending_accumulation,
                "effective_learning_rate": _effective_learning_rate(
                    optimizer_config,
                    base_learning_rate,
                    lr_step,
                ),
            }
        )
    return records


def _effective_learning_rate(
    optimizer_config: dict[str, Any],
    base_learning_rate: float,
    step: int,
) -> float:
    learning_rate = base_learning_rate
    warmup_steps = optimizer_config["warmup_steps"]
    decay_steps = optimizer_config["decay_steps"]
    if warmup_steps > 0:
        learning_rate *= min(1.0, step / warmup_steps)
    if decay_steps > 0 and step > warmup_steps:
        decay_step = min(step - warmup_steps, decay_steps)
        decay_fraction = decay_step / decay_steps
        learning_rate = learning_rate - (
            learning_rate - optimizer_config["min_learning_rate"]
        ) * decay_fraction
    return max(learning_rate, optimizer_config["min_learning_rate"])


def _adamw_contract(optimizer_config: dict[str, Any]) -> dict[str, Any] | None:
    if optimizer_config["optimizer"] != "adamw":
        return None
    return {
        "beta1": optimizer_config["beta1"],
        "beta2": optimizer_config["beta2"],
        "epsilon": optimizer_config["epsilon"],
        "weight_decay": optimizer_config["weight_decay"],
    }


def _expected_final_optimizer_state(
    *,
    parameter_manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    final = records[-1] if records else {}
    return {
        "update_count": final.get("update_count_after", 0),
        "pending_accumulation": final.get("pending_accumulation_after", 0),
        "param_count": parameter_manifest["parameter_count"],
    }


def _validate_step_record(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    summary = actual["optimizer_summary"]
    if expected["step"] != actual["step"]:
        raise ValueError("optimizer step contract step mismatch")
    if expected["update_count_after"] != summary["update_count"]:
        raise ValueError("optimizer step contract update_count mismatch")
    if expected["pending_accumulation_after"] != summary["pending_accumulation"]:
        raise ValueError("optimizer step contract pending_accumulation mismatch")
