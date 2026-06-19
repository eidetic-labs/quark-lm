"""Public transformer training parity artifact surface."""

from __future__ import annotations

from transformer_training_parity_fixture import (
    build_scalar_training_parity_fixture,
    validate_training_parity_fixture,
)
from transformer_optimizer_step_contract import (
    OPTIMIZER_STEP_CONTRACT_KIND,
    OPTIMIZER_STEP_CONTRACT_SCHEMA_VERSION,
    build_optimizer_step_contract,
    validate_optimizer_step_contract,
)
from transformer_gradient_accumulation_contract import (
    GRADIENT_ACCUMULATION_CONTRACT_KIND,
    GRADIENT_ACCUMULATION_CONTRACT_SCHEMA_VERSION,
    build_gradient_accumulation_contract,
    validate_gradient_accumulation_contract,
)
from transformer_training_parameter_manifest import (
    TRAINING_PARAMETER_MANIFEST_SCHEMA_VERSION,
    TRAINING_PARAMETER_ORDER,
    build_training_parameter_manifest,
    validate_training_parameter_manifest,
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
    "TRAINING_PARAMETER_MANIFEST_SCHEMA_VERSION",
    "TRAINING_PARAMETER_ORDER",
    "OPTIMIZER_STEP_CONTRACT_KIND",
    "OPTIMIZER_STEP_CONTRACT_SCHEMA_VERSION",
    "GRADIENT_ACCUMULATION_CONTRACT_KIND",
    "GRADIENT_ACCUMULATION_CONTRACT_SCHEMA_VERSION",
    "build_gradient_accumulation_contract",
    "build_optimizer_step_contract",
    "build_scalar_training_parity_fixture",
    "build_training_parameter_manifest",
    "build_training_parity_report",
    "validate_gradient_accumulation_contract",
    "validate_optimizer_step_contract",
    "validate_training_parameter_manifest",
    "validate_training_parity_fixture",
]
