"""Small PyTorch tensor helpers shared by transformer parity code."""

from __future__ import annotations

from typing import Any


def torch_tensor(torch: Any, value: Any, runtime: dict[str, Any]) -> Any:
    if _is_tensor_like(value):
        return value
    return torch.tensor(
        value,
        dtype=getattr(torch, runtime["dtype"]),
        device=runtime["device"],
    )


def torch_linear(
    inputs: Any,
    weights: list[list[float]],
    bias: list[float],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    return (inputs @ torch_tensor(torch, weights, runtime)) + torch_tensor(
        torch,
        bias,
        runtime,
    )


def torch_to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return [float(item) for item in value.tolist()]


def torch_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    raw_value = value.item() if hasattr(value, "item") else value.tolist()
    return float(raw_value)


def _is_tensor_like(value: Any) -> bool:
    return not isinstance(value, (list, int, float)) and hasattr(value, "tolist")
