"""Shared weight-decay exclusion predicate (single source of truth).

GPT-style AdamW excludes embeddings, normalization gains/biases, and all 1-D
bias vectors from weight decay, decaying only the multi-dimensional weight
matrices. This predicate is the ONE definition used by both the scalar optimizer
(as a per-element mask) and the torch AdamW (as a two-group split), so the two
backends cannot drift. ``no_decay == True`` means SKIP weight decay.
"""

from __future__ import annotations

from typing import Any


def is_no_decay(name: str, shape: list[int]) -> bool:
    """True if this tensor should be EXCLUDED from weight decay.

    Excludes every 1-D tensor (biases, norm gains), embeddings (matched by name,
    though 2-D), and any leaf ending in ``_gain``/``_bias`` (normalization
    parameters, including the dot-prefixed ``extra_layers.{i}.*`` copies -- the
    suffix is taken after the last '.'). Decays only the multi-dimensional weight
    matrices (wq/wk/wv/wo/w1/w2/wout/w_gate/...).
    """

    leaf = name.split(".")[-1]
    return (
        len(shape) <= 1
        or "embedding" in name
        or leaf.endswith("_gain")
        or leaf.endswith("_bias")
    )


def build_no_decay_mask(manifest: dict[str, Any]) -> list[bool]:
    """Per-element no-decay mask aligned with the flat scalar parameter order.

    Each manifest entry contributes ``count`` copies of its tensor-level flag, so
    the boolean list lines up element-for-element with the scalar optimizer's
    flattened parameters (the same order the manifest enumerates).
    """

    mask: list[bool] = []
    for entry in manifest["entries"]:
        excluded = is_no_decay(entry["name"], entry["shape"])
        mask.extend([excluded] * entry["count"])
    return mask


def model_no_decay_mask(model: Any) -> list[bool]:
    """Per-element no-decay mask for a model, from its parameter manifest.

    Convenience for production trainers: builds the manifest from the model's
    own weights/config so the scalar optimizer excludes the same tensors the
    torch two-group split does. Local imports avoid a module-load cycle.
    """

    from dataclasses import asdict

    from transformer_training_parameter_manifest import build_training_parameter_manifest

    manifest = build_training_parameter_manifest(
        weights=model.to_dict()["weights"], model_config=asdict(model.config)
    )
    return build_no_decay_mask(manifest)


def adamw_device_kwargs(device: str) -> dict[str, Any]:
    """Device-aware AdamW kernel selection (feature-complete on GPU/NPU/CPU).

    The fused AdamW kernel is CUDA-only and raises on CPU/MPS, so it is enabled
    only on CUDA. CPU/MPS (and any other device, e.g. an NPU backend) use torch's
    default path -- which the parity tests validate -- so the default behavior is
    unchanged. Pure + testable without a GPU.
    """

    if device == "cuda":
        return {"fused": True}
    return {}


def build_two_group_adamw(
    parameter_entries: list[dict[str, Any]],
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    torch: Any,
    device: str = "cpu",
) -> Any:
    """Build a torch AdamW that excludes the no-decay tensors from weight decay.

    Splits ``parameter_entries`` (each carrying name/shape/tensor) into a decay
    group and a no-decay group via the SAME ``is_no_decay`` predicate the scalar
    mask uses, so the two backends exclude identical tensors. At weight_decay=0
    the split is numerically inert (both groups decay nothing), preserving the
    pre-exclusion behavior. ``device`` selects the fused kernel on CUDA only.
    """

    decay = [e["tensor"] for e in parameter_entries if not is_no_decay(e["name"], e["shape"])]
    no_decay = [e["tensor"] for e in parameter_entries if is_no_decay(e["name"], e["shape"])]
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(
        groups, lr=learning_rate, betas=betas, eps=eps, **adamw_device_kwargs(device)
    )
