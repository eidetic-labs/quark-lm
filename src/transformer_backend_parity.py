"""Public transformer backend parity fixture surface."""

from __future__ import annotations

from transformer_backend_parity_fixture import build_scalar_backend_parity_fixture
from transformer_backend_parity_report import build_backend_parity_report
from transformer_backend_parity_schema import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    PARITY_FIXTURE_KIND,
    PARITY_REPORT_KIND,
    PARITY_SCHEMA_VERSION,
)
from transformer_backend_parity_validation import validate_backend_parity_fixture


__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "PARITY_FIXTURE_KIND",
    "PARITY_REPORT_KIND",
    "PARITY_SCHEMA_VERSION",
    "build_backend_parity_report",
    "build_scalar_backend_parity_fixture",
    "validate_backend_parity_fixture",
]
