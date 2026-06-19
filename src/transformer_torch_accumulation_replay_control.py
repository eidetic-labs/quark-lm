"""Control-flow probes for PyTorch accumulated-gradient replay plans."""

from __future__ import annotations

from typing import Any

from transformer_torch_gradient_clip import apply_torch_gradient_value_clip
from transformer_torch_optimizer_step_probe import (
    summarize_torch_optimizer_gradients,
)
from transformer_torch_tensor_ops import torch_to_float
from transformer_torch_training_loss import build_torch_training_loss_tensor


TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION = 1
TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS = (
    "accumulation_replay_control_recorded"
)


def build_torch_accumulation_replay_control_probe(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any] | None,
    torch: Any | None,
    runtime: dict[str, Any],
    replay_plan: dict[str, Any],
) -> dict[str, Any]:
    """Run replay-plan loss/backward control without claiming update parity."""

    if state is None or torch is None:
        return _not_run("pytorch training runtime is not ready")

    records = []
    for microstep in replay_plan["microsteps"]:
        record = _execute_microstep(
            fixture=fixture,
            state=state,
            torch=torch,
            runtime=runtime,
            microstep=microstep,
        )
        records.append(record)
        if record["status"] != "microstep_backward_recorded":
            break
    return {
        "schema_version": TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
        "status": _status(records, replay_plan),
        "reason": _reason(records, replay_plan),
        "case_id": replay_plan["case_id"],
        "planned_microstep_count": replay_plan["microstep_count"],
        "executed_microstep_count": len(records),
        "backward_pass_count": sum(
            1 for record in records if record["backward_called"]
        ),
        "optimizer_updates_applied": 0,
        "accumulated_gradient_parity_proven": False,
        "final_update_parity_proven": False,
        "microsteps": records,
    }


def _execute_microstep(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    microstep: dict[str, Any],
) -> dict[str, Any]:
    _clear_gradients(state)
    loss = build_torch_training_loss_tensor(
        fixture=fixture,
        state=state,
        torch=torch,
        runtime=runtime,
        context=microstep["context"],
        target=microstep["target"],
    )
    scaled_loss = loss * microstep["loss_scale"]
    if not callable(getattr(scaled_loss, "backward", None)):
        return _microstep_record(
            microstep=microstep,
            status="backward_unavailable",
            backward_called=False,
            loss_value=torch_to_float(loss),
            scaled_loss_value=torch_to_float(scaled_loss),
            clip_report=None,
            gradient_summary=summarize_torch_optimizer_gradients(
                state=state,
                contract=fixture["optimizer_step_contract"],
            ),
        )
    scaled_loss.backward()
    clip_report = None
    if microstep["clip_after_backward"]:
        clip_report = apply_torch_gradient_value_clip(
            torch=torch,
            state=state,
            clip_value=fixture["optimizer_step_contract"]["gradient_clip"][
                "value"
            ],
        )
    return _microstep_record(
        microstep=microstep,
        status="microstep_backward_recorded",
        backward_called=True,
        loss_value=torch_to_float(loss),
        scaled_loss_value=torch_to_float(scaled_loss),
        clip_report=clip_report,
        gradient_summary=summarize_torch_optimizer_gradients(
            state=state,
            contract=fixture["optimizer_step_contract"],
        ),
    )


def _microstep_record(
    *,
    microstep: dict[str, Any],
    status: str,
    backward_called: bool,
    loss_value: float,
    scaled_loss_value: float,
    clip_report: dict[str, Any] | None,
    gradient_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": microstep["step"],
        "status": status,
        "loss_scale": microstep["loss_scale"],
        "loss": loss_value,
        "scaled_loss": scaled_loss_value,
        "backward_called": backward_called,
        "clip_after_backward": microstep["clip_after_backward"],
        "clip_report": clip_report,
        "buffer_action": microstep["buffer_action"],
        "optimizer_step_after_microstep": microstep[
            "optimizer_step_after_microstep"
        ],
        "optimizer_step_applied": False,
        "gradient_summary": gradient_summary,
    }


def _status(records: list[dict[str, Any]], replay_plan: dict[str, Any]) -> str:
    if len(records) != replay_plan["microstep_count"]:
        return "accumulation_replay_control_incomplete"
    if any(record["status"] != "microstep_backward_recorded" for record in records):
        return "accumulation_replay_control_incomplete"
    return TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS


def _reason(records: list[dict[str, Any]], replay_plan: dict[str, Any]) -> str:
    if len(records) != replay_plan["microstep_count"]:
        return "microstep replay control did not reach every planned microstep"
    if any(record["status"] != "microstep_backward_recorded" for record in records):
        return "one or more planned backward passes could not run"
    return (
        "planned microstep backward control was recorded without optimizer "
        "updates"
    )


def _clear_gradients(state: dict[str, Any]) -> None:
    for parameter in state["parameters"]:
        tensor = parameter["tensor"]
        if hasattr(tensor, "grad"):
            tensor.grad = None


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
        "status": "not_run",
        "reason": reason,
    }
