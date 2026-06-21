"""Optional PyTorch runtime detection for transformer backend experiments."""

from __future__ import annotations

import random
from importlib import import_module
from typing import Any, Callable

from transformer_backend_policy import PYTORCH_BACKEND


TorchImporter = Callable[[str], Any]
TORCH_RUNTIME_KIND_PYTORCH = "pytorch"
TORCH_RUNTIME_KIND_TEST_DOUBLE = "test_double"
TORCH_RUNTIME_KIND_UNAVAILABLE = "unavailable"
ALLOWED_DTYPES = frozenset({"float64", "float32", "bfloat16", "float16"})


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


def configure_torch_runtime(
    torch: Any, runtime: dict[str, Any], *, seed: int | None = None
) -> None:
    """Validate runtime, disable TF32, and optionally seed the torch RNG.

    The determinism floor (Phase 0): reproducibility is a closed-world pillar and
    breaks silently on GPU (TF32 matmuls blow any float32 parity tolerance) or
    under unseeded torch randomness. This is a no-op for the current float64/CPU
    path -- today's training draws no torch randomness -- so it preserves bit-exact
    scalar parity while establishing the floor for the float32/on-device backend.
    """

    dtype = runtime.get("dtype")
    if dtype not in ALLOWED_DTYPES or not hasattr(torch, dtype):
        raise ValueError(f"unsupported runtime dtype: {dtype!r}")
    device = runtime.get("device")
    try:
        torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported runtime device: {device!r}") from exc
    _disable_tf32(torch)
    if seed is not None:
        torch.manual_seed(seed)


def epoch_shuffle_order(count: int, seed: int | None, epoch: int) -> list[int]:
    """Deterministic per-epoch permutation of example indices.

    Keyed by (seed + epoch) so each pass over the data sees a fresh but fully
    reproducible order -- breaks the fixed cyclic correlation of `step % len`
    that can let AdamW lock onto a phase across epochs, without sacrificing
    determinism (a closed-world pillar). seed=None pins to a fixed base so runs
    stay repeatable.
    """

    order = list(range(count))
    random.Random((seed if seed is not None else 0) + epoch).shuffle(order)
    return order


def grad_global_norm(params: list[Any], torch: Any) -> float:
    """Global L2 norm over all parameter gradients (0.0 when no grads are set).

    Update-health telemetry: a norm trending to 0 means dead gradients (the model
    has stopped learning); a norm exploding means an unstable step. Recorded per
    update so capacity scaling (Phase 6) can diagnose a tier that won't train
    instead of guessing. Observation-only -- reads .grad, never mutates it -- and
    measured pre-clip (raw gradient health, before any clip masks an explosion),
    so scalar parity is untouched.
    """

    total: Any = None
    for parameter in params:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        squared = (grad.detach() ** 2).sum()
        total = squared if total is None else total + squared
    if total is None:
        return 0.0
    return float(total.sqrt().cpu())


def _disable_tf32(torch: Any) -> None:
    """Force full-precision float32 matmuls so the parity band is meaningful on GPU."""

    backends = getattr(torch, "backends", None)
    matmul = getattr(getattr(backends, "cuda", None), "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = False
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        cudnn.allow_tf32 = False


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
