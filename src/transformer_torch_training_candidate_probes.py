"""Probe artifact assembly for PyTorch training candidates."""

from __future__ import annotations

from typing import Any

from transformer_torch_runtime import TorchImporter
from transformer_torch_optimizer_step_probe import (
    build_torch_optimizer_step_probe,
)
from transformer_torch_optimizer_step_execution import (
    build_torch_optimizer_step_execution_probe,
)
from transformer_torch_training_backward_probe import (
    build_torch_training_backward_probe,
)
from transformer_torch_training_loss_probe import (
    build_torch_training_initial_loss_probe,
)
from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS
from transformer_torch_training_replay_probes import (
    build_torch_training_replay_probes,
)
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
    optimizer_step_probe = build_torch_optimizer_step_probe(
        fixture=fixture,
        state=state,
        backward_probe=backward_probe,
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
        **build_torch_training_replay_probes(
            fixture=fixture,
            importer=importer,
            readiness=readiness,
            runtime=runtime,
        ),
        "optimizer_step_probe": optimizer_step_probe,
        "optimizer_step_execution_probe": _optimizer_step_execution_probe(
            fixture=fixture,
            importer=importer,
            state=state,
            optimizer_step_probe=optimizer_step_probe,
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


def _optimizer_step_execution_probe(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    state: dict[str, Any] | None,
    optimizer_step_probe: dict[str, Any],
) -> dict[str, Any]:
    torch = importer("torch") if state is not None else None
    return build_torch_optimizer_step_execution_probe(
        fixture=fixture,
        state=state,
        optimizer_step_probe=optimizer_step_probe,
        torch=torch,
    )
