"""Public optional PyTorch backend experiment surface."""

from __future__ import annotations

from transformer_torch_parity_candidate import (
    TORCH_PARITY_CANDIDATE_KIND,
    TORCH_PARITY_CANDIDATE_SCHEMA_VERSION,
    TORCH_PARITY_IMPLEMENTATION_STATUS,
    build_torch_backend_parity_candidate,
)
from transformer_torch_adamw_expected_update import (
    TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS,
    TORCH_ADAMW_EXPECTED_UPDATE_SCHEMA_VERSION,
    build_torch_adamw_expected_update,
)
from transformer_torch_gradient_clip import (
    TORCH_GRADIENT_CLIP_APPLIED_STATUS,
    TORCH_GRADIENT_CLIP_SCHEMA_VERSION,
    apply_torch_gradient_value_clip,
)
from transformer_torch_gradient_accumulation import (
    TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS,
    TORCH_GRADIENT_ACCUMULATION_SCHEMA_VERSION,
    build_torch_gradient_accumulation_report,
)
from transformer_torch_gradient_snapshot import (
    TORCH_GRADIENT_SNAPSHOT_SCHEMA_VERSION,
    snapshot_torch_gradients,
)
from transformer_torch_replay_gradient_comparison import (
    TORCH_REPLAY_GRADIENT_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_GRADIENT_MATCHED_STATUS,
    TORCH_REPLAY_GRADIENT_MISMATCH_STATUS,
    build_torch_replay_gradient_comparison,
)
from transformer_torch_accumulation_readiness import (
    TORCH_ACCUMULATION_PENDING_STATUS,
    TORCH_ACCUMULATION_READINESS_SCHEMA_VERSION,
    TORCH_ACCUMULATION_READY_STATUS,
    build_torch_accumulation_readiness,
)
from transformer_torch_accumulation_replay_plan import (
    TORCH_ACCUMULATION_REPLAY_PENDING_STATUS,
    TORCH_ACCUMULATION_REPLAY_PLAN_SCHEMA_VERSION,
    build_torch_accumulation_replay_plan,
)
from transformer_torch_accumulation_replay_control import (
    TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS,
    TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
    build_torch_accumulation_replay_control_probe,
)
from transformer_torch_runtime import torch_runtime_status
from transformer_torch_optimizer_step_probe import (
    TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION,
    TORCH_OPTIMIZER_STEP_READY_STATUS,
    build_torch_optimizer_step_probe,
    summarize_torch_optimizer_gradients,
)
from transformer_torch_optimizer_step_execution import (
    TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS,
    TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
    build_torch_optimizer_step_execution_probe,
)
from transformer_torch_parameter_mutation import (
    TORCH_PARAMETER_MUTATION_NOT_OBSERVED_STATUS,
    TORCH_PARAMETER_MUTATION_OBSERVED_STATUS,
    TORCH_PARAMETER_MUTATION_SCHEMA_VERSION,
    build_torch_parameter_mutation_report,
    snapshot_torch_parameters,
)
from transformer_torch_parameter_signature_comparison import (
    TORCH_PARAMETER_SIGNATURE_COMPARISON_SCHEMA_VERSION,
    TORCH_PARAMETER_SIGNATURE_MATCHED_STATUS,
    TORCH_PARAMETER_SIGNATURE_MISMATCH_STATUS,
    build_torch_parameter_signature_comparison,
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
    "TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS",
    "TORCH_ADAMW_EXPECTED_UPDATE_SCHEMA_VERSION",
    "TORCH_GRADIENT_CLIP_APPLIED_STATUS",
    "TORCH_GRADIENT_CLIP_SCHEMA_VERSION",
    "TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS",
    "TORCH_GRADIENT_ACCUMULATION_SCHEMA_VERSION",
    "TORCH_GRADIENT_SNAPSHOT_SCHEMA_VERSION",
    "TORCH_REPLAY_GRADIENT_COMPARISON_SCHEMA_VERSION",
    "TORCH_REPLAY_GRADIENT_MATCHED_STATUS",
    "TORCH_REPLAY_GRADIENT_MISMATCH_STATUS",
    "TORCH_ACCUMULATION_PENDING_STATUS",
    "TORCH_ACCUMULATION_READINESS_SCHEMA_VERSION",
    "TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS",
    "TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION",
    "TORCH_ACCUMULATION_REPLAY_PENDING_STATUS",
    "TORCH_ACCUMULATION_REPLAY_PLAN_SCHEMA_VERSION",
    "TORCH_ACCUMULATION_READY_STATUS",
    "TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION",
    "TORCH_OPTIMIZER_STEP_READY_STATUS",
    "TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS",
    "TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION",
    "TORCH_PARAMETER_MUTATION_NOT_OBSERVED_STATUS",
    "TORCH_PARAMETER_MUTATION_OBSERVED_STATUS",
    "TORCH_PARAMETER_MUTATION_SCHEMA_VERSION",
    "TORCH_PARAMETER_SIGNATURE_COMPARISON_SCHEMA_VERSION",
    "TORCH_PARAMETER_SIGNATURE_MATCHED_STATUS",
    "TORCH_PARAMETER_SIGNATURE_MISMATCH_STATUS",
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
    "build_torch_adamw_expected_update",
    "build_torch_backend_parity_candidate",
    "apply_torch_gradient_value_clip",
    "snapshot_torch_gradients",
    "build_torch_replay_gradient_comparison",
    "build_torch_accumulation_readiness",
    "build_torch_accumulation_replay_control_probe",
    "build_torch_accumulation_replay_plan",
    "build_torch_gradient_accumulation_report",
    "build_torch_optimizer_step_probe",
    "build_torch_optimizer_step_execution_probe",
    "build_torch_training_initial_loss_probe",
    "build_torch_parameter_mutation_report",
    "build_torch_parameter_signature_comparison",
    "build_torch_training_backward_probe",
    "build_torch_training_parity_candidate",
    "build_torch_training_readiness",
    "build_torch_training_state",
    "summarize_torch_training_state",
    "summarize_torch_optimizer_gradients",
    "snapshot_torch_parameters",
    "torch_training_weights_from_state",
    "torch_runtime_status",
    "validate_torch_training_state_summary",
]
