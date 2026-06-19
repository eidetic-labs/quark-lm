"""Gradient-accumulation semantics for scalar training parity contracts."""

from __future__ import annotations

from typing import Any


GRADIENT_ACCUMULATION_CONTRACT_SCHEMA_VERSION = 1
GRADIENT_ACCUMULATION_CONTRACT_KIND = "gradient_accumulation_contract"


def build_gradient_accumulation_contract(
    *,
    optimizer_config: dict[str, Any],
) -> dict[str, Any]:
    """Describe how scalar training reduces accumulated microstep gradients."""

    steps = int(optimizer_config["gradient_accumulation_steps"])
    if steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    clip_value = float(optimizer_config["gradient_clip"])
    requires_microstep_clipping = clip_value > 0.0
    return {
        "schema_version": GRADIENT_ACCUMULATION_CONTRACT_SCHEMA_VERSION,
        "kind": GRADIENT_ACCUMULATION_CONTRACT_KIND,
        "steps": steps,
        "reduction": "mean",
        "gradient_source": "clipped_microstep_gradients",
        "buffering": "sum_then_divide_by_pending_accumulation",
        "divisor": "pending_accumulation",
        "requires_microstep_clipping": requires_microstep_clipping,
        "pytorch_equivalence": {
            "requires_clipped_gradient_buffer": requires_microstep_clipping,
            "native_loss_scaling_sufficient": not requires_microstep_clipping,
            "loss_scale_if_no_microstep_clipping": 1.0 / steps,
        },
    }


def validate_gradient_accumulation_contract(
    contract: dict[str, Any],
    *,
    steps: int,
    gradient_clip: float,
) -> None:
    """Validate gradient-accumulation semantics against optimizer settings."""

    if contract.get("schema_version") != GRADIENT_ACCUMULATION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported gradient accumulation contract schema_version")
    if contract.get("kind") != GRADIENT_ACCUMULATION_CONTRACT_KIND:
        raise ValueError(f"kind must be {GRADIENT_ACCUMULATION_CONTRACT_KIND}")
    if contract.get("steps") != steps:
        raise ValueError("gradient accumulation contract steps mismatch")
    if contract.get("reduction") != "mean":
        raise ValueError("gradient accumulation contract reduction must be mean")
    if contract.get("gradient_source") != "clipped_microstep_gradients":
        raise ValueError("gradient accumulation contract gradient_source mismatch")
    expected_clipping = gradient_clip > 0.0
    if contract.get("requires_microstep_clipping") != expected_clipping:
        raise ValueError("gradient accumulation contract clipping mismatch")
    equivalence = contract.get("pytorch_equivalence", {})
    if equivalence.get("requires_clipped_gradient_buffer") != expected_clipping:
        raise ValueError("gradient accumulation contract buffer mismatch")
    if equivalence.get("native_loss_scaling_sufficient") != (not expected_clipping):
        raise ValueError("gradient accumulation contract loss-scaling mismatch")
