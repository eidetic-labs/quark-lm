"""Probe artifact assembly for PyTorch training candidates."""

from __future__ import annotations

from typing import Any

from transformer_torch_runtime import TorchImporter
from transformer_torch_training_backward_probe import (
    build_torch_training_backward_probe,
)
from transformer_torch_training_loss_probe import (
    build_torch_training_initial_loss_probe,
)
from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS
from transformer_torch_training_state import (
    build_torch_training_state,
    summarize_torch_training_state,
)


def build_torch_training_probe_artifacts(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    readiness: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Build JSON-safe probe artifacts for a PyTorch training candidate."""

    state = _training_state(
        fixture=fixture,
        importer=importer,
        readiness=readiness,
        runtime=runtime,
    )
    backward_probe = _backward_probe(
        fixture=fixture,
        importer=importer,
        runtime=runtime,
        state=state,
    )
    return {
        "training_state": _training_state_summary(state),
        "initial_loss_probe": _initial_loss_probe(
            fixture=fixture,
            importer=importer,
            runtime=runtime,
            state=state,
        ),
        "backward_probe": backward_probe,
        "optimizer_step_probe": _optimizer_step_probe(
            fixture=fixture,
            backward_probe=backward_probe,
        ),
    }


def _training_state(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    readiness: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any] | None:
    if readiness["status"] != TORCH_TRAINING_READY_STATUS:
        return None
    return build_torch_training_state(
        fixture=fixture,
        torch=importer("torch"),
        runtime=runtime,
    )


def _training_state_summary(state: dict[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {
            "status": "not_built",
            "reason": "pytorch training runtime is not ready",
        }
    return {
        "status": "built",
        **summarize_torch_training_state(state),
    }


def _initial_loss_probe(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    runtime: dict[str, Any],
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    if state is None:
        return {
            "status": "not_run",
            "reason": "pytorch training runtime is not ready",
        }
    return build_torch_training_initial_loss_probe(
        fixture=fixture,
        torch=importer("torch"),
        runtime=runtime,
        state=state,
    )


def _backward_probe(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    runtime: dict[str, Any],
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    if state is None:
        return {
            "status": "not_run",
            "reason": "pytorch training runtime is not ready",
        }
    return build_torch_training_backward_probe(
        fixture=fixture,
        torch=importer("torch"),
        runtime=runtime,
        state=state,
    )


def _optimizer_step_probe(
    *,
    fixture: dict[str, Any],
    backward_probe: dict[str, Any],
) -> dict[str, Any]:
    if backward_probe.get("status") != "gradients_available":
        return {
            "status": "not_run",
            "reason": "pytorch gradients are not available",
        }
    contract = fixture["optimizer_step_contract"]
    return {
        "status": "pending_optimizer_implementation",
        "reason": "pytorch optimizer step parity is not implemented yet",
        "optimizer": contract["optimizer"],
        "expected_update_count": contract["expected_final_optimizer_state"][
            "update_count"
        ],
        "expected_step_count": len(contract["expected_step_records"]),
    }
