"""Replay-specific probe assembly for PyTorch training candidates."""

from __future__ import annotations

from typing import Any

from transformer_torch_accumulation_replay_control import (
    build_torch_accumulation_replay_control_probe,
)
from transformer_torch_accumulation_replay_plan import (
    build_torch_accumulation_replay_plan,
)
from transformer_torch_replay_buffer_comparison import (
    build_torch_replay_buffer_comparison,
)
from transformer_torch_replay_final_evaluation import (
    build_torch_replay_final_evaluation,
)
from transformer_torch_replay_update_comparison import (
    build_torch_replay_update_comparison,
)
from transformer_torch_runtime import TorchImporter
from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS
from transformer_torch_training_state import build_torch_training_state


def build_torch_training_replay_probes(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    readiness: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Build replay-plan, buffer, update, and final-eval artifacts."""

    replay_plan = build_torch_accumulation_replay_plan(fixture=fixture)
    control = _replay_control_probe(
        fixture=fixture,
        importer=importer,
        readiness=readiness,
        replay_plan=replay_plan,
        runtime=runtime,
    )
    buffer = build_torch_replay_buffer_comparison(
        fixture=fixture,
        replay_control_probe=control,
    )
    update = _replay_update_comparison(
        fixture=fixture,
        importer=importer,
        readiness=readiness,
        replay_control_probe=control,
        buffer_comparison=buffer,
        runtime=runtime,
    )
    return {
        "accumulation_replay_plan": replay_plan,
        "accumulation_replay_control_probe": control,
        "accumulation_replay_buffer_comparison": buffer,
        "accumulation_replay_update_comparison": update,
        "accumulation_replay_final_evaluation": _final_evaluation(
            fixture=fixture,
            importer=importer,
            readiness=readiness,
            replay_control_probe=control,
            buffer_comparison=buffer,
            update_comparison=update,
            runtime=runtime,
        ),
    }


def _replay_control_probe(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    readiness: dict[str, Any],
    replay_plan: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    state = _training_state(
        fixture=fixture,
        importer=importer,
        readiness=readiness,
        runtime=runtime,
    )
    return build_torch_accumulation_replay_control_probe(
        fixture=fixture,
        state=state,
        torch=importer("torch") if state is not None else None,
        runtime=runtime,
        replay_plan=replay_plan,
    )


def _replay_update_comparison(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    readiness: dict[str, Any],
    replay_control_probe: dict[str, Any],
    buffer_comparison: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    state = _training_state(
        fixture=fixture,
        importer=importer,
        readiness=readiness,
        runtime=runtime,
    )
    return build_torch_replay_update_comparison(
        fixture=fixture,
        state=state,
        torch=importer("torch") if state is not None else None,
        runtime=runtime,
        replay_control_probe=replay_control_probe,
        buffer_comparison=buffer_comparison,
    )


def _final_evaluation(
    *,
    fixture: dict[str, Any],
    importer: TorchImporter,
    readiness: dict[str, Any],
    replay_control_probe: dict[str, Any],
    buffer_comparison: dict[str, Any],
    update_comparison: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    state = _training_state(
        fixture=fixture,
        importer=importer,
        readiness=readiness,
        runtime=runtime,
    )
    return build_torch_replay_final_evaluation(
        fixture=fixture,
        state=state,
        torch=importer("torch") if state is not None else None,
        runtime=runtime,
        replay_control_probe=replay_control_probe,
        buffer_comparison=buffer_comparison,
        update_comparison=update_comparison,
    )


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
