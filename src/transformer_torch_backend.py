"""Public optional PyTorch backend experiment surface."""

from __future__ import annotations

from transformer_torch_parity_candidate import (
    TORCH_PARITY_CANDIDATE_KIND,
    TORCH_PARITY_CANDIDATE_SCHEMA_VERSION,
    TORCH_PARITY_IMPLEMENTATION_STATUS,
    build_torch_backend_parity_candidate,
)
from transformer_torch_runtime import torch_runtime_status
from transformer_torch_training_candidate import (
    TORCH_TRAINING_IMPLEMENTATION_STATUS,
    TORCH_TRAINING_PARITY_CANDIDATE_KIND,
    TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION,
    build_torch_training_parity_candidate,
)


__all__ = [
    "TORCH_PARITY_CANDIDATE_KIND",
    "TORCH_PARITY_CANDIDATE_SCHEMA_VERSION",
    "TORCH_PARITY_IMPLEMENTATION_STATUS",
    "TORCH_TRAINING_IMPLEMENTATION_STATUS",
    "TORCH_TRAINING_PARITY_CANDIDATE_KIND",
    "TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION",
    "build_torch_backend_parity_candidate",
    "build_torch_training_parity_candidate",
    "torch_runtime_status",
]
