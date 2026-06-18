"""Public transformer training parity artifact surface."""

from __future__ import annotations

from transformer_training_parity_fixture import (
    build_scalar_training_parity_fixture,
    validate_training_parity_fixture,
)
from transformer_training_parity_report import build_training_parity_report
from transformer_training_parity_schema import (
    TRAINING_PARITY_FIXTURE_KIND,
    TRAINING_PARITY_REPORT_KIND,
    TRAINING_PARITY_SCHEMA_VERSION,
)


__all__ = [
    "TRAINING_PARITY_FIXTURE_KIND",
    "TRAINING_PARITY_REPORT_KIND",
    "TRAINING_PARITY_SCHEMA_VERSION",
    "build_scalar_training_parity_fixture",
    "build_training_parity_report",
    "validate_training_parity_fixture",
]
