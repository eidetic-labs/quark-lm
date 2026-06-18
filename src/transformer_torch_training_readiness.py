"""Runtime readiness checks for optional PyTorch training parity."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from transformer_torch_runtime import TorchImporter
from transformer_training_parameter_manifest import (
    validate_training_parameter_manifest,
)


TORCH_TRAINING_READINESS_SCHEMA_VERSION = 1
TORCH_TRAINING_READY_STATUS = "ready"
TORCH_TRAINING_PENDING_STATUS = "pending"
TORCH_TRAINING_BLOCKED_STATUS = "blocked"


def build_torch_training_readiness(
    *,
    fixture: dict[str, Any],
    runtime: dict[str, Any],
    importer: TorchImporter = import_module,
) -> dict[str, Any]:
    """Report whether an optional PyTorch runtime can attempt training parity."""

    checks = [
        _runtime_available_check(runtime),
        _dtype_available_check(runtime),
        _parameter_manifest_check(fixture),
    ]
    if runtime.get("available") and runtime.get("dtype_available"):
        torch = importer("torch")
        checks.extend(
            [
                _capability_check("torch_tensor", getattr(torch, "tensor", None)),
                _autograd_check(torch, runtime),
                _adamw_check(torch),
            ]
        )
    return {
        "schema_version": TORCH_TRAINING_READINESS_SCHEMA_VERSION,
        "status": _readiness_status(checks),
        "checks": checks,
        "summary": _summary(checks),
    }


def _runtime_available_check(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "runtime_available",
        "passed": bool(runtime.get("available")),
        "error": runtime.get("error"),
    }


def _dtype_available_check(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "dtype_available",
        "passed": bool(runtime.get("dtype_available")),
        "dtype": runtime.get("dtype"),
    }


def _parameter_manifest_check(fixture: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_training_parameter_manifest(
            fixture["parameter_manifest"],
            optimizer_state=fixture["training_case"]["optimizer_state"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "name": "parameter_manifest",
            "passed": False,
            "error": str(exc),
        }
    return {
        "name": "parameter_manifest",
        "passed": True,
        "parameter_count": fixture["parameter_manifest"]["parameter_count"],
    }


def _capability_check(name: str, capability: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": callable(capability),
    }


def _autograd_check(torch: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    if getattr(torch, "autograd", None) is None:
        return {"name": "autograd", "passed": False}
    try:
        probe = torch.tensor(
            0.0,
            dtype=getattr(torch, runtime["dtype"]),
            device=runtime["device"],
            requires_grad=True,
        )
    except TypeError:
        return {
            "name": "autograd",
            "passed": False,
            "error": "tensor constructor does not accept requires_grad",
        }
    return {
        "name": "autograd",
        "passed": callable(getattr(probe, "backward", None)),
    }


def _adamw_check(torch: Any) -> dict[str, Any]:
    optimizer_module = getattr(torch, "optim", None)
    return {
        "name": "adamw_optimizer",
        "passed": callable(getattr(optimizer_module, "AdamW", None)),
    }


def _readiness_status(checks: list[dict[str, Any]]) -> str:
    if all(check["passed"] for check in checks):
        return TORCH_TRAINING_READY_STATUS
    if not checks[0]["passed"]:
        return TORCH_TRAINING_BLOCKED_STATUS
    return TORCH_TRAINING_PENDING_STATUS


def _summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "check_count": len(checks),
        "passed_check_count": sum(1 for check in checks if check["passed"]),
        "failed_checks": [
            check["name"] for check in checks if check["passed"] is not True
        ],
    }
