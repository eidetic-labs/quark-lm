"""PyTorch accumulation-runtime readiness for training parity evidence."""

from __future__ import annotations

from typing import Any


TORCH_ACCUMULATION_READINESS_SCHEMA_VERSION = 1
TORCH_ACCUMULATION_READY_STATUS = "accumulation_runtime_ready"
TORCH_ACCUMULATION_PENDING_STATUS = "accumulation_runtime_pending"


def build_torch_accumulation_readiness(
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Report runtime pieces needed for scalar-equivalent accumulation."""

    accumulation = contract["gradient_accumulation"]
    requirements = _requirements(contract=contract, accumulation=accumulation)
    missing = [
        requirement["name"]
        for requirement in requirements
        if requirement["required"] and requirement["status"] == "missing"
    ]
    return {
        "schema_version": TORCH_ACCUMULATION_READINESS_SCHEMA_VERSION,
        "status": _status(missing),
        "reason": _reason(missing),
        "accumulation_steps": contract["gradient_accumulation_steps"],
        "requires_replayed_backward_passes": _requires_replay(contract),
        "requires_clipped_gradient_buffer": accumulation["pytorch_equivalence"][
            "requires_clipped_gradient_buffer"
        ],
        "native_loss_scaling_sufficient": accumulation["pytorch_equivalence"][
            "native_loss_scaling_sufficient"
        ],
        "missing_requirements": missing,
        "requirements": requirements,
    }


def _requirements(
    *,
    contract: dict[str, Any],
    accumulation: dict[str, Any],
) -> list[dict[str, Any]]:
    replay_required = _requires_replay(contract)
    buffer_required = accumulation["pytorch_equivalence"][
        "requires_clipped_gradient_buffer"
    ]
    native_scaling = accumulation["pytorch_equivalence"][
        "native_loss_scaling_sufficient"
    ]
    return [
        _requirement(
            name="replay_backward_per_microstep",
            required=replay_required,
            reason="scalar evidence spans more than one optimizer microstep",
        ),
        _requirement(
            name="scale_loss_by_accumulation_steps",
            required=replay_required and native_scaling,
            reason="unclipped accumulation can use native loss scaling",
        ),
        _requirement(
            name="clipped_gradient_buffer",
            required=buffer_required,
            reason="scalar optimizer averages clipped microstep gradients",
        ),
        _requirement(
            name="mean_gradient_reduction",
            required=replay_required or buffer_required,
            reason="scalar optimizer divides buffered gradients before AdamW",
        ),
    ]


def _requirement(
    *,
    name: str,
    required: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "status": "missing" if required else "not_required",
        "reason": reason,
    }


def _requires_replay(contract: dict[str, Any]) -> bool:
    return (
        contract["gradient_accumulation_steps"] > 1
        or len(contract["expected_step_records"]) > 1
    )


def _status(missing: list[str]) -> str:
    if missing:
        return TORCH_ACCUMULATION_PENDING_STATUS
    return TORCH_ACCUMULATION_READY_STATUS


def _reason(missing: list[str]) -> str:
    if missing:
        return "PyTorch accumulation parity is waiting on runtime support"
    return "no additional accumulation runtime support is required"
