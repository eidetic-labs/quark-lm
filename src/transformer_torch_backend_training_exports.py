"""Training public exports for the optional PyTorch backend experiment."""

from __future__ import annotations

from transformer_torch_training_backward_probe import (
    TORCH_TRAINING_BACKWARD_PROBE_SCHEMA_VERSION,
    build_torch_training_backward_probe,
)
from transformer_torch_training_candidate import (
    TORCH_TRAINING_PARITY_CANDIDATE_KIND,
    TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION,
    TORCH_TRAINING_REPLAY_PARITY_STATUS,
    TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS,
    build_torch_training_parity_candidate,
)
from transformer_torch_training_candidate_validation import (
    REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS,
    validate_torch_training_parity_candidate,
)
from transformer_torch_training_loss_probe import (
    TORCH_TRAINING_LOSS_PROBE_SCHEMA_VERSION,
    build_torch_training_initial_loss_probe,
)
from transformer_torch_training_parity_attempt_audit import (
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
    build_torch_training_parity_attempt_audit,
)
from transformer_torch_training_parity_attempt_audit_validation import (
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES,
    validate_torch_training_parity_attempt_audit,
)
from transformer_torch_training_parity_attempt_hashes import (
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
    build_torch_runtime_report_hash,
    build_torch_training_attempt_payload_hash,
    build_torch_training_parity_attempt_hashes,
)
from transformer_torch_training_parity_attempt_reader import (
    TORCH_TRAINING_PARITY_ATTEMPT_FILES,
    load_torch_training_parity_attempt_artifact_set,
)
from transformer_torch_training_parity_attempt_requirement_validation import (
    validate_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_parity_attempt_summary_validation import (
    validate_torch_training_parity_attempt_summaries,
)
from transformer_torch_training_parity_attempt_requirements import (
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS,
    build_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION,
    TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
    build_torch_training_backend_promotion_gate,
)
from transformer_torch_training_promotion_gate_validation import (
    validate_torch_training_backend_promotion_gate,
)
from transformer_torch_training_readiness import (
    TORCH_TRAINING_BLOCKED_STATUS,
    TORCH_TRAINING_PENDING_STATUS,
    TORCH_TRAINING_READINESS_SCHEMA_VERSION,
    TORCH_TRAINING_READY_STATUS,
    build_torch_training_readiness,
)
from transformer_torch_training_readiness_validation import (
    TORCH_TRAINING_READINESS_BASE_CHECKS,
    TORCH_TRAINING_READINESS_CHECK_CATALOGS,
    TORCH_TRAINING_READINESS_RUNTIME_CHECKS,
    validate_torch_training_readiness,
)
from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
    TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION,
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
    build_torch_training_replay_parity_gate,
)
from transformer_torch_training_state import (
    TORCH_TRAINING_STATE_SCHEMA_VERSION,
    build_torch_training_state,
    summarize_torch_training_state,
    torch_training_weights_from_state,
    validate_torch_training_state_summary,
)


__all__ = [
    "TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM",
    "TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS",
    "TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS",
    "TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION",
    "TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES",
    "TORCH_TRAINING_BACKWARD_PROBE_SCHEMA_VERSION",
    "TORCH_TRAINING_BLOCKED_STATUS",
    "TORCH_TRAINING_LOSS_PROBE_SCHEMA_VERSION",
    "TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS",
    "TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND",
    "TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES",
    "TORCH_TRAINING_PARITY_ATTEMPT_FILES",
    "TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES",
    "TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND",
    "TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS",
    "TORCH_TRAINING_PARITY_CANDIDATE_KIND",
    "TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION",
    "TORCH_TRAINING_PENDING_STATUS",
    "TORCH_TRAINING_READINESS_BASE_CHECKS",
    "TORCH_TRAINING_READINESS_CHECK_CATALOGS",
    "TORCH_TRAINING_READINESS_RUNTIME_CHECKS",
    "TORCH_TRAINING_READINESS_SCHEMA_VERSION",
    "TORCH_TRAINING_READY_STATUS",
    "TORCH_TRAINING_REPLAY_BLOCKED_STATUS",
    "TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION",
    "TORCH_TRAINING_REPLAY_MATCHED_STATUS",
    "TORCH_TRAINING_REPLAY_PARITY_STATUS",
    "TORCH_TRAINING_REPLAY_PENDING_STATUS",
    "TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS",
    "TORCH_TRAINING_STATE_SCHEMA_VERSION",
    "REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS",
    "build_torch_training_backend_promotion_gate",
    "build_torch_training_backward_probe",
    "build_torch_training_initial_loss_probe",
    "build_torch_runtime_report_hash",
    "build_torch_training_attempt_payload_hash",
    "build_torch_training_parity_attempt_audit",
    "build_torch_training_parity_attempt_hashes",
    "build_torch_training_parity_attempt_requirements",
    "build_torch_training_parity_candidate",
    "build_torch_training_readiness",
    "build_torch_training_replay_parity_gate",
    "build_torch_training_state",
    "load_torch_training_parity_attempt_artifact_set",
    "summarize_torch_training_state",
    "torch_training_weights_from_state",
    "validate_torch_training_parity_attempt_requirements",
    "validate_torch_training_parity_attempt_audit",
    "validate_torch_training_parity_attempt_summaries",
    "validate_torch_training_backend_promotion_gate",
    "validate_torch_training_parity_candidate",
    "validate_torch_training_readiness",
    "validate_torch_training_state_summary",
]
