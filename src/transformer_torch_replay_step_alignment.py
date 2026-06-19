"""Step-alignment checks for PyTorch replay evidence."""

from __future__ import annotations

from typing import Any


TORCH_REPLAY_STEP_ALIGNMENT_SCHEMA_VERSION = 1
TORCH_REPLAY_STEP_ALIGNMENT_MATCHED_STATUS = "replay_step_alignment_matched"
TORCH_REPLAY_STEP_ALIGNMENT_MISMATCH_STATUS = "replay_step_alignment_mismatch"


def build_torch_replay_step_alignment(
    *,
    replay_control_probe: dict[str, Any],
    scalar_step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify replay microsteps cover scalar step evidence exactly."""

    replay_steps = [
        microstep.get("step")
        for microstep in replay_control_probe.get("microsteps", [])
    ]
    scalar_steps = [record.get("step") for record in scalar_step_records]
    passed = replay_steps == scalar_steps
    return {
        "schema_version": TORCH_REPLAY_STEP_ALIGNMENT_SCHEMA_VERSION,
        "status": (
            TORCH_REPLAY_STEP_ALIGNMENT_MATCHED_STATUS
            if passed
            else TORCH_REPLAY_STEP_ALIGNMENT_MISMATCH_STATUS
        ),
        "passed": passed,
        "replay_step_count": len(replay_steps),
        "scalar_step_count": len(scalar_steps),
        "replay_steps": replay_steps,
        "scalar_steps": scalar_steps,
        "reason": _reason(passed),
    }


def _reason(passed: bool) -> str:
    if passed:
        return "replay microsteps exactly match scalar training steps"
    return "replay microsteps do not exactly match scalar training steps"
