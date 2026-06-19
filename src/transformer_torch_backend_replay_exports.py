"""Replay and update public exports for the PyTorch backend experiment."""

from __future__ import annotations

from transformer_torch_accumulation_readiness import (
    TORCH_ACCUMULATION_PENDING_STATUS,
    TORCH_ACCUMULATION_READINESS_SCHEMA_VERSION,
    TORCH_ACCUMULATION_READY_STATUS,
    build_torch_accumulation_readiness,
)
from transformer_torch_accumulation_replay_control import (
    TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS,
    TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
    build_torch_accumulation_replay_control_probe,
)
from transformer_torch_accumulation_replay_plan import (
    TORCH_ACCUMULATION_REPLAY_PENDING_STATUS,
    TORCH_ACCUMULATION_REPLAY_PLAN_SCHEMA_VERSION,
    build_torch_accumulation_replay_plan,
)
from transformer_torch_adamw_expected_update import (
    TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS,
    TORCH_ADAMW_EXPECTED_UPDATE_SCHEMA_VERSION,
    build_torch_adamw_expected_update,
)
from transformer_torch_gradient_accumulation import (
    TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS,
    TORCH_GRADIENT_ACCUMULATION_SCHEMA_VERSION,
    build_torch_gradient_accumulation_report,
)
from transformer_torch_gradient_clip import (
    TORCH_GRADIENT_CLIP_APPLIED_STATUS,
    TORCH_GRADIENT_CLIP_SCHEMA_VERSION,
    apply_torch_gradient_value_clip,
)
from transformer_torch_gradient_snapshot import (
    TORCH_GRADIENT_SNAPSHOT_SCHEMA_VERSION,
    snapshot_torch_gradients,
)
from transformer_torch_optimizer_step_execution import (
    TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS,
    TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION,
    build_torch_optimizer_step_execution_probe,
)
from transformer_torch_optimizer_step_probe import (
    TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION,
    TORCH_OPTIMIZER_STEP_READY_STATUS,
    build_torch_optimizer_step_probe,
    summarize_torch_optimizer_gradients,
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
from transformer_torch_replay_buffer_comparison import (
    TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_BUFFER_MATCHED_STATUS,
    TORCH_REPLAY_BUFFER_MISMATCH_STATUS,
    TORCH_REPLAY_BUFFER_NOT_RUN_STATUS,
    build_torch_replay_buffer_comparison,
)
from transformer_torch_replay_checkpoint_compatibility import (
    TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
    TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS,
    TORCH_REPLAY_CHECKPOINT_MISMATCH_STATUS,
    TORCH_REPLAY_CHECKPOINT_NOT_RUN_STATUS,
    build_torch_replay_checkpoint_compatibility,
)
from transformer_torch_replay_final_evaluation import (
    TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
    TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS,
    TORCH_REPLAY_FINAL_EVAL_MISMATCH_STATUS,
    TORCH_REPLAY_FINAL_EVAL_NOT_RUN_STATUS,
    build_torch_replay_final_evaluation,
)
from transformer_torch_replay_gradient_comparison import (
    TORCH_REPLAY_GRADIENT_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_GRADIENT_MATCHED_STATUS,
    TORCH_REPLAY_GRADIENT_MISMATCH_STATUS,
    build_torch_replay_gradient_comparison,
)
from transformer_torch_replay_update_comparison import (
    TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_UPDATE_MATCHED_STATUS,
    TORCH_REPLAY_UPDATE_MISMATCH_STATUS,
    TORCH_REPLAY_UPDATE_NOT_RUN_STATUS,
    build_torch_replay_update_comparison,
)


__all__ = [
    "TORCH_ACCUMULATION_PENDING_STATUS",
    "TORCH_ACCUMULATION_READINESS_SCHEMA_VERSION",
    "TORCH_ACCUMULATION_READY_STATUS",
    "TORCH_ACCUMULATION_REPLAY_CONTROL_RECORDED_STATUS",
    "TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION",
    "TORCH_ACCUMULATION_REPLAY_PENDING_STATUS",
    "TORCH_ACCUMULATION_REPLAY_PLAN_SCHEMA_VERSION",
    "TORCH_ADAMW_EXPECTED_UPDATE_BUILT_STATUS",
    "TORCH_ADAMW_EXPECTED_UPDATE_SCHEMA_VERSION",
    "TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS",
    "TORCH_GRADIENT_ACCUMULATION_SCHEMA_VERSION",
    "TORCH_GRADIENT_CLIP_APPLIED_STATUS",
    "TORCH_GRADIENT_CLIP_SCHEMA_VERSION",
    "TORCH_GRADIENT_SNAPSHOT_SCHEMA_VERSION",
    "TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS",
    "TORCH_OPTIMIZER_STEP_EXECUTION_SCHEMA_VERSION",
    "TORCH_OPTIMIZER_STEP_PROBE_SCHEMA_VERSION",
    "TORCH_OPTIMIZER_STEP_READY_STATUS",
    "TORCH_PARAMETER_MUTATION_NOT_OBSERVED_STATUS",
    "TORCH_PARAMETER_MUTATION_OBSERVED_STATUS",
    "TORCH_PARAMETER_MUTATION_SCHEMA_VERSION",
    "TORCH_PARAMETER_SIGNATURE_COMPARISON_SCHEMA_VERSION",
    "TORCH_PARAMETER_SIGNATURE_MATCHED_STATUS",
    "TORCH_PARAMETER_SIGNATURE_MISMATCH_STATUS",
    "TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION",
    "TORCH_REPLAY_BUFFER_MATCHED_STATUS",
    "TORCH_REPLAY_BUFFER_MISMATCH_STATUS",
    "TORCH_REPLAY_BUFFER_NOT_RUN_STATUS",
    "TORCH_REPLAY_CHECKPOINT_MATCHED_STATUS",
    "TORCH_REPLAY_CHECKPOINT_MISMATCH_STATUS",
    "TORCH_REPLAY_CHECKPOINT_NOT_RUN_STATUS",
    "TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION",
    "TORCH_REPLAY_FINAL_EVAL_MATCHED_STATUS",
    "TORCH_REPLAY_FINAL_EVAL_MISMATCH_STATUS",
    "TORCH_REPLAY_FINAL_EVAL_NOT_RUN_STATUS",
    "TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION",
    "TORCH_REPLAY_GRADIENT_COMPARISON_SCHEMA_VERSION",
    "TORCH_REPLAY_GRADIENT_MATCHED_STATUS",
    "TORCH_REPLAY_GRADIENT_MISMATCH_STATUS",
    "TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION",
    "TORCH_REPLAY_UPDATE_MATCHED_STATUS",
    "TORCH_REPLAY_UPDATE_MISMATCH_STATUS",
    "TORCH_REPLAY_UPDATE_NOT_RUN_STATUS",
    "apply_torch_gradient_value_clip",
    "build_torch_accumulation_readiness",
    "build_torch_accumulation_replay_control_probe",
    "build_torch_accumulation_replay_plan",
    "build_torch_adamw_expected_update",
    "build_torch_gradient_accumulation_report",
    "build_torch_optimizer_step_execution_probe",
    "build_torch_optimizer_step_probe",
    "build_torch_parameter_mutation_report",
    "build_torch_parameter_signature_comparison",
    "build_torch_replay_buffer_comparison",
    "build_torch_replay_checkpoint_compatibility",
    "build_torch_replay_final_evaluation",
    "build_torch_replay_gradient_comparison",
    "build_torch_replay_update_comparison",
    "snapshot_torch_gradients",
    "snapshot_torch_parameters",
    "summarize_torch_optimizer_gradients",
]
