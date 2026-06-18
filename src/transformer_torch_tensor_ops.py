"""Small PyTorch tensor helpers shared by transformer parity code."""

from __future__ import annotations

from typing import Any


def torch_tensor(torch: Any, value: Any, runtime: dict[str, Any]) -> Any:
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
