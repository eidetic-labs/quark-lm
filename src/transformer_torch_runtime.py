"""Optional PyTorch runtime detection for transformer backend experiments."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from transformer_backend_policy import PYTORCH_BACKEND


TorchImporter = Callable[[str], Any]
TORCH_RUNTIME_KIND_PYTORCH = "pytorch"
TORCH_RUNTIME_KIND_TEST_DOUBLE = "test_double"
TORCH_RUNTIME_KIND_UNAVAILABLE = "unavailable"


def torch_runtime_status(
    *,
    importer: TorchImporter = import_module,
    requested_device: str = "cpu",
    requested_dtype: str = "float32",
) -> dict[str, Any]:
    """Describe PyTorch availability without making it a required dependency."""

    try:
        torch = importer("torch")
    except Exception as exc:
        return {
            "backend": PYTORCH_BACKEND,
            "available": False,
            "runtime_kind": TORCH_RUNTIME_KIND_UNAVAILABLE,
            "version": None,
            "requested_device": requested_device,
            "device": "cpu",
            "requested_dtype": requested_dtype,
            "dtype": requested_dtype,
            "dtype_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    available_devices = _available_devices(torch)
    device = _resolve_device(requested_device, available_devices)
    return {
        "backend": PYTORCH_BACKEND,
        "available": True,
        "runtime_kind": _runtime_kind(torch),
        "version": str(getattr(torch, "__version__", "unknown")),
        "requested_device": requested_device,
        "device": device,
        "available_devices": available_devices,
        "requested_dtype": requested_dtype,
        "dtype": requested_dtype,
        "dtype_available": hasattr(torch, requested_dtype),
        "error": None,
    }


def _available_devices(torch: Any) -> list[str]:
    devices = ["cpu"]
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and _safe_bool_call(getattr(cuda, "is_available", None)):
        devices.append("cuda")
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and _safe_bool_call(getattr(mps, "is_available", None)):
        devices.append("mps")
    return devices


def _runtime_kind(torch: Any) -> str:
    version = str(getattr(torch, "__version__", ""))
    if version.startswith("fake-"):
        return TORCH_RUNTIME_KIND_TEST_DOUBLE
    return TORCH_RUNTIME_KIND_PYTORCH


def _resolve_device(requested_device: str, available_devices: list[str]) -> str:
    requested = requested_device.strip().lower()
    if requested == "auto":
        for device in ("cuda", "mps", "cpu"):
            if device in available_devices:
                return device
    if requested in available_devices:
        return requested
    return "cpu"


def _safe_bool_call(callback: Any) -> bool:
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception:
        return False
