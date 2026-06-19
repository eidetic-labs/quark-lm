"""Core public exports for the optional PyTorch backend experiment."""

from __future__ import annotations

from transformer_torch_parity_candidate import (
    TORCH_PARITY_CANDIDATE_KIND,
    TORCH_PARITY_CANDIDATE_SCHEMA_VERSION,
    TORCH_PARITY_IMPLEMENTATION_STATUS,
    build_torch_backend_parity_candidate,
)
from transformer_torch_runtime import (
    TORCH_RUNTIME_KIND_PYTORCH,
    TORCH_RUNTIME_KIND_TEST_DOUBLE,
    TORCH_RUNTIME_KIND_UNAVAILABLE,
    torch_runtime_status,
)
from transformer_torch_runtime_report import (
    TORCH_RUNTIME_REPORT_KIND,
    build_torch_runtime_report,
    write_torch_runtime_report,
)
from transformer_torch_runtime_report_validation import (
    TORCH_RUNTIME_REPORT_CHECKS,
    TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE,
    TORCH_RUNTIME_REPORT_STATUSES,
    validate_torch_runtime_report,
)


__all__ = [
    "TORCH_PARITY_CANDIDATE_KIND",
    "TORCH_PARITY_CANDIDATE_SCHEMA_VERSION",
    "TORCH_PARITY_IMPLEMENTATION_STATUS",
    "TORCH_RUNTIME_KIND_PYTORCH",
    "TORCH_RUNTIME_KIND_TEST_DOUBLE",
    "TORCH_RUNTIME_KIND_UNAVAILABLE",
    "TORCH_RUNTIME_REPORT_CHECKS",
    "TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE",
    "TORCH_RUNTIME_REPORT_KIND",
    "TORCH_RUNTIME_REPORT_STATUSES",
    "build_torch_backend_parity_candidate",
    "build_torch_runtime_report",
    "torch_runtime_status",
    "validate_torch_runtime_report",
    "write_torch_runtime_report",
]
