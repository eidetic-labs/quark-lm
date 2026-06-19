"""Public optional PyTorch backend experiment surface."""

from __future__ import annotations

from transformer_torch_parity_candidate import (
    TORCH_PARITY_CANDIDATE_KIND,
    TORCH_PARITY_CANDIDATE_SCHEMA_VERSION,
    TORCH_PARITY_IMPLEMENTATION_STATUS,
    build_torch_backend_parity_candidate,
)
from transformer_torch_runtime import torch_runtime_status
from transformer_torch_optimizer_step_probe import (
    TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION,
    TORCH_OPTIMIZER_STEP_READY_STATUS,
    build_torch_optimizer_step_probe,
    summarize_torch_optimizer_gradients,
)
from transformer_torch_training_candidate import (
    TORCH_TRAINING_IMPLEMENTATION_STATUS,
    TORCH_TRAINING_PARITY_CANDIDATE_KIND,
    TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION,
    TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
    build_torch_training_parity_candidate,
)
from transformer_torch_training_readiness import (
    TORCH_TRAINING_BLOCKED_STATUS,
    TORCH_TRAINING_PENDING_STATUS,
    TORCH_TRAINING_READINESS_SCHEMA_VERSION,
    TORCH_TRAINING_READY_STATUS,
    build_torch_training_readiness,
)
from transformer_torch_training_loss_probe import (
    TORCH_TRAINING_LOSS_PROBE_SCHEMA_VERSION,
    build_torch_training_initial_loss_probe,
)
from transformer_torch_training_backward_probe import (
    TORCH_TRAINING_BACKWARD_PROBE_SCHEMA_VERSION,
    build_torch_training_backward_probe,
)
from transformer_torch_training_state import (
    TORCH_TRAINING_STATE_SCHEMA_VERSION,
    build_torch_training_state,
    summarize_torch_training_state,
    torch_training_weights_from_state,
    validate_torch_training_state_summary,
)


__all__ = [
    "TORCH_PARITY_CANDIDATE_KIND",
    "TORCH_PARITY_CANDIDATE_SCHEMA_VERSION",
    "TORCH_PARITY_IMPLEMENTATION_STATUS",
    "TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION",
    "TORCH_OPTIMIZER_STEP_READY_STATUS",
    "TORCH_TRAINING_IMPLEMENTATION_STATUS",
    "TORCH_TRAINING_BLOCKED_STATUS",
    "TORCH_TRAINING_PENDING_STATUS",
    "TORCH_TRAINING_PARITY_CANDIDATE_KIND",
    "TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION",
    "TORCH_TRAINING_READINESS_SCHEMA_VERSION",
    "TORCH_TRAINING_READY_STATUS",
    "TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS",
    "TORCH_TRAINING_LOSS_PROBE_SCHEMA_VERSION",
    "TORCH_TRAINING_BACKWARD_PROBE_SCHEMA_VERSION",
    "TORCH_TRAINING_STATE_SCHEMA_VERSION",
    "build_torch_backend_parity_candidate",
    "build_torch_optimizer_step_probe",
    "build_torch_training_initial_loss_probe",
    "build_torch_training_backward_probe",
    "build_torch_training_parity_candidate",
    "build_torch_training_readiness",
    "build_torch_training_state",
    "summarize_torch_training_state",
    "summarize_torch_optimizer_gradients",
    "torch_training_weights_from_state",
    "torch_runtime_status",
    "validate_torch_training_state_summary",
]
