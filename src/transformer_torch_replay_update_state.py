"""Apply replay-buffer gradients to a fresh PyTorch training state."""

from __future__ import annotations

from typing import Any

from transformer_torch_parameter_mutation import snapshot_torch_parameters


TORCH_REPLAY_UPDATE_STATE_APPLIED_STATUS = "replay_update_state_applied"
TORCH_REPLAY_UPDATE_STATE_NOT_RUN_STATUS = "replay_update_state_not_run"


def apply_torch_replay_update_to_state(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any] | None,
    torch: Any | None,
    runtime: dict[str, Any],
    replay_control_probe: dict[str, Any],
    buffer_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Apply replayed accumulated gradients to ``state`` in place."""

    if state is None or torch is None:
        return _not_run("pytorch training runtime is not ready")
    if not buffer_comparison.get("passed"):
        return _not_run("replay buffer parity has not passed")
    if fixture["optimizer_step_contract"]["optimizer"] != "adamw":
        return _not_run("only AdamW replay updates are currently supported")

    updates = _applied_updates(
        fixture=fixture,
        replay_control_probe=replay_control_probe,
    )
    if len(updates) != 1:
        return _not_run("replay update comparison currently supports one update")
    update = updates[0]
    if update["update_count_after"] != 1:
        return _not_run("replay update comparison requires zero prior moments")

    optimizer = _build_adamw_optimizer(
        torch=torch,
        state=state,
        contract=fixture["optimizer_step_contract"],
    )
    _set_optimizer_learning_rate(optimizer, update["effective_learning_rate"])
    parameters_before = snapshot_torch_parameters(state)
    _assign_flat_gradients(
        state=state,
        torch=torch,
        runtime=runtime,
        gradients=update["gradients"],
    )
    optimizer.step()
    optimizer.zero_grad()
    parameters_after = snapshot_torch_parameters(state)
    return {
        "status": TORCH_REPLAY_UPDATE_STATE_APPLIED_STATUS,
        "case_id": replay_control_probe["case_id"],
        "optimizer": "adamw",
        "applied_update_count": 1,
        "effective_learning_rate": update["effective_learning_rate"],
        "gradient_signature": _signature(update["gradients"]),
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
    }


def _applied_updates(
    *,
    fixture: dict[str, Any],
    replay_control_probe: dict[str, Any],
) -> list[dict[str, Any]]:
    buffer = [0.0 for _index in range(fixture["parameter_manifest"]["parameter_count"])]
    updates = []
    for microstep, scalar_record in zip(
        replay_control_probe["microsteps"],
        fixture["training_case"]["step_records"],
    ):
        evidence = scalar_record["optimizer_gradient_evidence"]
        buffer = _add_vectors(buffer, _snapshot_values(microstep["gradient_snapshot"]))
        if evidence["update_applied"]:
            divisor = evidence["pending_accumulation_before"] + 1
            updates.append(
                {
                    "step": scalar_record["step"],
                    "update_count_after": evidence["update_count_after"],
                    "effective_learning_rate": evidence["learning_rate"],
                    "gradients": [value / divisor for value in buffer],
                }
            )
            buffer = [0.0 for _index in range(len(buffer))]
    return updates


def _assign_flat_gradients(
    *,
    state: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    gradients: list[float],
) -> None:
    for parameter in state["parameters"]:
        values = gradients[parameter["index_start"]:parameter["index_end"]]
        parameter["tensor"].grad = torch.tensor(
            _reshape(values, parameter["shape"]),
            dtype=getattr(torch, runtime["dtype"]),
            device=runtime["device"],
        )


def _reshape(values: list[float], shape: list[int]) -> Any:
    if not shape:
        return values[0]
    stride = _element_count(shape[1:])
    return [
        _reshape(values[index * stride:(index + 1) * stride], shape[1:])
        for index in range(shape[0])
    ]


def _element_count(shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _build_adamw_optimizer(
    *,
    torch: Any,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Any:
    config = contract["adamw"]
    return torch.optim.AdamW(
        [parameter["tensor"] for parameter in state["parameters"]],
        lr=contract["base_learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["epsilon"],
        weight_decay=config["weight_decay"],
    )


def _set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for group in getattr(optimizer, "param_groups", []):
        group["lr"] = learning_rate


def _snapshot_values(snapshot: dict[str, Any]) -> list[float]:
    return [
        value
        for parameter in snapshot["parameters"]
        for value in parameter["values"]
    ]


def _add_vectors(left: list[float], right: list[float]) -> list[float]:
    return [left_value + right_value for left_value, right_value in zip(left, right)]


def _signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "status": TORCH_REPLAY_UPDATE_STATE_NOT_RUN_STATUS,
        "reason": reason,
    }
